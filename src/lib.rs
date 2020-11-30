//! crate docs

#![no_std]
#![warn(clippy::all, clippy::pedantic, clippy::cargo, missing_docs, missing_doc_code_examples)]
#![allow(clippy::clippy::must_use_candidate)]

extern crate alloc;

use core::{
    convert::TryInto,
    iter::FromIterator,
    ptr::NonNull,
    ops::{Index, IndexMut},
    hash::{Hash, Hasher},
    fmt::{Debug, Formatter, Result as FmtResult},
    sync::atomic::{AtomicU8, AtomicUsize, Ordering},
};
use alloc::{vec::Vec, boxed::Box};
use segment::Segment;

mod segment;

/// We keep this as a [`u8`] to intentionally limit the amount of possible
/// segments.
const SEGMENTS: u8 = 32;

/// A **dy**namically-allocated, **co**ncurrent **vec**tor.
///
/// [`DycoVec`]s are thread-safe vectors that support random access, iteration,
/// and tail pushing in an `O(1)`, wait-free\* manner, *and that's it*! It's
/// possible, given `&mut self`, to clear `self` and mutate elements, but to do
/// so without `&mut self` requires wrapping `T` in a type such as
/// [`Mutex`](std::sync::Mutex).
///
/// # Technical Details
///
/// The technical details of [`DycoVec`]s are something you're likely to find
/// boring and uninteresting. Read below if you wish, but beware!
///
/// ## Memory Use
///
/// Storage is allocated in lazily-allocated individually-contiguous chunks.
/// This means that, given a first chunk capacity of `n`, pushing less than `n`
/// (but at least one) elements uses the same memory as pushing `n` elements,
/// while pushing `n + 1` elements makes the [`DycoVec`] now have two segments:
/// the first chunk, and a larger second chunk.
///
/// ## Almost Wait-free
///
/// The [`DycoVec`] is almost always wait-free to push on to, and is otherwise
/// still lock-free. The only exception to wait-freedom is when a segment has
/// filled and a new one is being allocated. This will result in every thread
/// simultaneously attempting to enter a specific critical section. The first
/// to do so will allocate the segment. After finishing, all other threads
/// vying for the lock will enter the critical section as appropriate, notice
/// the segment is already allocated, and immediately unlock. So, the maximum
/// hold-up will be O(`n`) (`n` being the number of threads). But, as a
/// reminder, this whole parade only happens when the [`DycoVec`] has yet to
/// be allocated.
///
/// The only other time the lock is locked is during cloning as to prevent an
/// incomplete allocation from providing bad data for the clone.
pub struct DycoVec<T> {
    segs: [Segment<T>; SEGMENTS as usize],
    cur_seg: AtomicU8,
    len: AtomicUsize,
}

impl<T: Clone> Clone for DycoVec<T> {
    fn clone(&self) -> Self {
        if Self::T_ZST {
            DycoVec {
                segs: Self::DEFAULT_SEGS,
                cur_seg: AtomicU8::new(0),
                len: AtomicUsize::new(self.len.load(Ordering::Acquire)),
            }
        } else {
            let segs = self.segs.clone();
            let cur_seg = segs
                .iter()
                .enumerate()
                .find_map(|(i, s)|
                    if s.is_empty() {
                        #[allow(clippy::cast_possible_truncation)]
                        Some(i as u8)
                    } else {
                        None
                    }
                )
                .unwrap_or(SEGMENTS);
            let mut len = 0;

            for seg in &segs[0..cur_seg as usize] {
                len += seg.len();
            }

            DycoVec {segs, cur_seg: AtomicU8::new(cur_seg), len: AtomicUsize::new(len)}
        }
    }

    fn clone_from(&mut self, source: &Self) {
        for i in 0..SEGMENTS as usize {
            self.segs[i].clone_from(&source.segs[i]);
        }

        *self.cur_seg.get_mut() = source.cur_seg.load(Ordering::Acquire);
        *self.len.get_mut() = source.len.load(Ordering::Acquire);
    }
}

impl<T: Hash> Hash for DycoVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.segs.hash(state);
    }
}

impl<T> From<DycoVec<T>> for Vec<T> {
    fn from(dv: DycoVec<T>) -> Self {
        let mut v = Vec::with_capacity(dv.len.into_inner());

        for i in 0..=dv.cur_seg.into_inner() {
            v.extend(
                // SAFE: `dv` is consumed
                unsafe { (&dv.segs[i as usize] as *const Segment<T>).read() }
            )
        }

        v
    }
}

impl<'a, T: 'a> From<&'a DycoVec<T>> for Vec<&'a T> {
    fn from(dv: &'a DycoVec<T>) -> Self {
        dv.segs.iter().flat_map(|s| s.iter()).collect()
    }
}

impl<T> IntoIterator for DycoVec<T> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        Vec::into_iter(self.into())
    }
}

impl<'a, T: 'a> IntoIterator for &'a DycoVec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> Extend<T> for DycoVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for i in iter {
            self.push(i);
        }
    }
}

impl<T> Default for DycoVec<T> {
    fn default() -> Self {
        DycoVec::new()
    }
}

impl<T> FromIterator<T> for DycoVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        DycoVec::from_boxed_slice(iter.into_iter().collect::<Vec<_>>().into_boxed_slice())
    }
}

impl<T> Index<usize> for DycoVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        //self.get(index).unwrap_or_else(|| oob())
        let out = self.get(index);
        out.unwrap_or_else(|| oob())
    }
}

impl<T> IndexMut<usize> for DycoVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        //self.get_mut(index).unwrap_or_else(|| oob())
        let out = self.get_mut(index);
        out.unwrap_or_else(|| oob())
    }
}

impl<T: Debug> Debug for DycoVec<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.debug_list()
            .entries(self)
            .finish()
    }
}

impl<T: PartialEq> PartialEq for DycoVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.segs == other.segs
    }
}

impl<T: Eq> Eq for DycoVec<T> {}

impl<T: PartialOrd> PartialOrd for DycoVec<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.segs.partial_cmp(&other.segs)
    }
}

impl<T: Ord> Ord for DycoVec<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.segs.cmp(&other.segs)
    }
}

impl<T> DycoVec<T> {
    // Scary, but the copying is actually intentional.
    #[allow(clippy::declare_interior_mutable_const)]
    const DEFAULT_SEGS: [Segment<T>; SEGMENTS as usize] = [
        Segment::new_null(CAPACITIES[0]),
        Segment::new_null(CAPACITIES[1]),
        Segment::new_null(CAPACITIES[2]),
        Segment::new_null(CAPACITIES[3]),
        Segment::new_null(CAPACITIES[4]),
        Segment::new_null(CAPACITIES[5]),
        Segment::new_null(CAPACITIES[6]),
        Segment::new_null(CAPACITIES[7]),
        Segment::new_null(CAPACITIES[8]),
        Segment::new_null(CAPACITIES[9]),
        Segment::new_null(CAPACITIES[10]),
        Segment::new_null(CAPACITIES[11]),
        Segment::new_null(CAPACITIES[12]),
        Segment::new_null(CAPACITIES[13]),
        Segment::new_null(CAPACITIES[14]),
        Segment::new_null(CAPACITIES[15]),
        Segment::new_null(CAPACITIES[16]),
        Segment::new_null(CAPACITIES[17]),
        Segment::new_null(CAPACITIES[18]),
        Segment::new_null(CAPACITIES[19]),
        Segment::new_null(CAPACITIES[20]),
        Segment::new_null(CAPACITIES[21]),
        Segment::new_null(CAPACITIES[22]),
        Segment::new_null(CAPACITIES[23]),
        Segment::new_null(CAPACITIES[24]),
        Segment::new_null(CAPACITIES[25]),
        Segment::new_null(CAPACITIES[26]),
        Segment::new_null(CAPACITIES[27]),
        Segment::new_null(CAPACITIES[28]),
        Segment::new_null(CAPACITIES[29]),
        Segment::new_null(CAPACITIES[30]),
        Segment::new_null(CAPACITIES[31]),
    ];
    
    const T_ZST: bool = core::mem::size_of::<T>() == 0;

    /// Create a new [`DycoVec`] with zero length.
    ///
    /// The [`DycoVec`] will no allocate memory until elements are pushed onto
    /// it.
    pub const fn new() -> Self {
        DycoVec {
            segs: Self::DEFAULT_SEGS,
            cur_seg: AtomicU8::new(0),
            len: AtomicUsize::new(0),
        }
    }
    
    /// Create a new [`DycoVec`] from a [box](Box)ed slice.
    #[allow(clippy::clippy::boxed_local)]
    pub fn from_boxed_slice(mut s: Box<[T]>) -> Self {
        let mut seg_id = 1;
        let len = s.len();
        let (mut current, mut leftover) = s.split_at_mut(CAPACITIES[0].min(len));
        let mut segs = Self::DEFAULT_SEGS;
        segs[0] = Segment::from_slice(
            &mut core::mem::take(&mut current),
            CAPACITIES[0],
        );

        while !leftover.is_empty() {
            let (c, l) = leftover.split_at_mut(CAPACITIES[seg_id].min(leftover.len()));
            current = c;
            leftover = l;
            segs[seg_id] = Segment::from_slice(
                &mut core::mem::take(&mut current),
                CAPACITIES[seg_id],
            );
            seg_id += 1;
        }

        DycoVec {
            segs,
            cur_seg: AtomicU8::new((seg_id - 1).try_into().unwrap()),
            len: AtomicUsize::new(len),
        }
    }

    /// Index `self` by `index`
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        if Self::T_ZST {
            NonNull::<T>::dangling().as_ptr().as_ref().unwrap()
        } else {
            let (seg_id, local_index) = id_to_seg_lid(index);
            
            self.segs[seg_id as usize].get_unchecked(local_index)
        }
    }

    /// Mutably index `self` by `index`
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        if Self::T_ZST {
            NonNull::<T>::dangling().as_ptr().as_mut().unwrap()
        } else {
            let (seg_id, local_index) = id_to_seg_lid(index);

            self.segs[seg_id as usize].get_unchecked_mut(local_index)
        }
    }

    /// Index `self` by `index`
    pub fn get(&self, index: usize) -> Option<&T> {
        if Self::T_ZST {
            if index < self.len() {
                // SAFE: we get some arbitrary pointer back for a ZST, whose
                // pointers are neither actually read nor written to. ie, it
                // doesn't matter what pointer is given, as long as it's
                // aligned and nonnull.
                Some(unsafe { self.get_unchecked(index) })
            } else {
                None
            }
        } else {
            let (seg_id, local_index) = id_to_seg_lid(index);
            
            if self.segs[seg_id as usize].is_initialized() {
                self.segs[seg_id as usize].get(local_index)
            } else {
                oob()
            }
        }
    }

    /// Mutably index `self` by `index`
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if Self::T_ZST {
            if index < self.len() {
                // SAFE: we get some arbitrary pointer back for a ZST, whose
                // pointers are neither actually read nor written to. ie, it
                // doesn't matter what pointer is given, as long as it's
                // aligned and nonnull.
                Some(unsafe { self.get_unchecked_mut(index) })
            } else {
                None
            }
        } else {
            let (seg_id, local_index) = id_to_seg_lid(index);
            
            if self.segs[seg_id as usize].is_initialized() {
                self.segs[seg_id as usize].get_mut(local_index)
            } else {
                oob()
            }
        }
    }

    /// Clear all elements in `self`.
    ///
    /// Drops all elements stored in `self` without losing the allocated space.
    pub fn clear(&mut self) {
        *self.len.get_mut() = 0;
        *self.cur_seg.get_mut() = 0;

        for seg in &mut self.segs {
            seg.clear()
        }
    }

    /// Push an element to the end of `self` and return the new index.
    pub fn push(&self, t: T) -> usize {
        self.len.fetch_add(1, Ordering::Release);

        let mut seg_id = self.cur_seg.load(Ordering::Acquire);
        let tref = &mut (&t as *const _);
        let mut out;

        while {out = self.segs[seg_id as usize].push(tref); out.is_none()} {
            self.cur_seg.fetch_max(seg_id + 1, Ordering::Release);
            seg_id = self.cur_seg.load(Ordering::Acquire);
        }

        core::mem::forget(t);
        seg_lid_to_id(seg_id, out.unwrap())
    }

    /// Get the number of elements in `self`.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Returns `true` if the [`DycoVec`] contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create an immutable iterator over the elements of `self`.
    pub const fn iter(&self) -> Iter<'_, T> {
        Iter {
            dv: self,
            index: 0,
        }
    }

    /// Consumes the [`DycoVec`], returning a [`Vec`] containing the elements.
    pub fn into_vec(self) -> Vec<T> {
        self.into()
    }
}

#[cold]
fn oob() -> ! {
    panic!("access out of bounds")
}

const INITIAL_EXP: u32 = 8;
const CAP_BASE: usize = 2;

const fn seg_n_capacity_const(seg_id: usize) -> usize {
    pow_usize(CAP_BASE, INITIAL_EXP as usize + seg_id)
}

const fn pow_usize(b: usize, mut e: usize) -> usize {
    if e == 0 {
        0
    } else {
        let mut o = b;

        #[allow(clippy::clippy::while_immutable_condition)]
        while {e -= 1; e} > 0 {
            o *= b;
        }

        o
    }
}

// https://www.wolframalpha.com/input/?i=sum+p%5E%28k%2Bn%29%2C+for+n%3D0+to+m-1
const MAX_ID: usize = {
    let p_k = pow_usize(CAP_BASE, INITIAL_EXP as usize);
    let p_m_1 = pow_usize(CAP_BASE, SEGMENTS as usize) - 1;
    (p_k * p_m_1) / (CAP_BASE - 1) as usize - 1
};

fn id_to_seg_lid(mut index: usize) -> (u8, usize) {
    debug_assert!(index < MAX_ID);
    let mut seg_id: u8 = 0;
    
    for &cap in &CAPACITIES {
        if index >= cap {
            index -= cap;
            seg_id += 1;
        } else {
            return (seg_id, index);
        }
    }

    oob()
}

fn seg_lid_to_id(mut seg_id: u8, mut lid: usize) -> usize {
    debug_assert!(seg_id < SEGMENTS);

    while 0 < seg_id {
        lid += CAPACITIES[seg_id as usize - 1];
        seg_id -= 1;
    }

    lid
}

const CAPACITIES: [usize; SEGMENTS as usize] = [
    seg_n_capacity_const(0),
    seg_n_capacity_const(1),
    seg_n_capacity_const(2),
    seg_n_capacity_const(3),
    seg_n_capacity_const(4),
    seg_n_capacity_const(5),
    seg_n_capacity_const(6),
    seg_n_capacity_const(7),
    seg_n_capacity_const(8),
    seg_n_capacity_const(9),
    seg_n_capacity_const(10),
    seg_n_capacity_const(11),
    seg_n_capacity_const(12),
    seg_n_capacity_const(13),
    seg_n_capacity_const(14),
    seg_n_capacity_const(15),
    seg_n_capacity_const(16),
    seg_n_capacity_const(17),
    seg_n_capacity_const(18),
    seg_n_capacity_const(19),
    seg_n_capacity_const(20),
    seg_n_capacity_const(21),
    seg_n_capacity_const(22),
    seg_n_capacity_const(23),
    seg_n_capacity_const(24),
    seg_n_capacity_const(25),
    seg_n_capacity_const(26),
    seg_n_capacity_const(27),
    seg_n_capacity_const(28),
    seg_n_capacity_const(29),
    seg_n_capacity_const(30),
    seg_n_capacity_const(31),
];

/// Immutable iter over [`DycoVec`]s.
///
/// Created by the [iter](DycoVec::iter()) method.
#[derive(Clone, Copy)]
pub struct Iter<'a, T: 'a> {
    dv: &'a DycoVec<T>,
    index: usize,
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let r = self.dv.get(self.index);
        self.index += 1;
        r
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.dv.len(), Some(MAX_ID - self.index))
    }

    fn count(self) -> usize {
        self.dv.len() - self.index
    }

    fn last(mut self) -> Option<Self::Item> {
        self.index = self.index.max(self.dv.len() - 1);
        self.next()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n;
        self.next()
    }

    fn for_each<F>(self, mut f: F)
    where
        F: FnMut(Self::Item),
    {
        for i in self.index..self.dv.len() {
            f(&self.dv[i]);
        }
    }
}

#[cfg(test)]
mod test {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn len() {
        let dv = DycoVec::new();
        assert!(dv.is_empty());
        assert_eq!(dv.len(), 0);
        dv.push(1);
        assert_eq!(dv.len(), 1);
        dv.push(2);
        assert_eq!(dv.len(), 2);

        for i in 0..=CAPACITIES[0] + CAPACITIES[1] {
            dv.push(i);
            assert_eq!(dv.len(), i + 3);
        }
    }

    #[test]
    fn len_zst() {
        let dv = DycoVec::new();
        assert!(dv.is_empty());
        assert_eq!(dv.len(), 0);
        dv.push(());
        assert_eq!(dv.len(), 1);
        dv.push(());
        assert_eq!(dv.len(), 2);

        for i in 0..=CAPACITIES[0] + CAPACITIES[1] {
            dv.push(());
            assert_eq!(dv.len(), i + 3);
        }
    }

    #[test]
    fn clear() {
        let mut dv = DycoVec::new();

        for i in 0..=CAPACITIES[0] + CAPACITIES[1] {
            dv.push(i);
        }

        assert_eq!(dv.len(), CAPACITIES[0] + CAPACITIES[1] + 1);
        dv.clear();
        assert!(dv.is_empty());

        for i in 0..CAPACITIES[0] {
            dv.push(i);
        }

        assert_eq!(dv.len(), CAPACITIES[0]);
    }

    #[test]
    fn clear_zst() {
        let mut dv = DycoVec::new();

        for _ in 0..=CAPACITIES[0] + CAPACITIES[1] {
            dv.push(());
        }

        assert_eq!(dv.len(), CAPACITIES[0] + CAPACITIES[1] + 1);
        dv.clear();
        assert!(dv.is_empty());
        assert_eq!(*dv.cur_seg.get_mut(), 0);

        for _ in 0..CAPACITIES[0] {
            dv.push(());
        }

        assert_eq!(dv.len(), CAPACITIES[0]);
    }

    #[test]
    fn index() {
        let dv = DycoVec::new();

        for i in 0..CAPACITIES[0] {
            dv.push(i);
            assert_eq!(dv[i], i);
            assert_eq!(unsafe { *dv.get_unchecked(i) }, i);
        }

        for i in 0..CAPACITIES[1] {
            dv.push(i);
            assert_eq!(dv[i + CAPACITIES[0]], i);
        }
    }

    #[test]
    fn index_zst() {
        let dv = DycoVec::new();

        for i in 0..(CAPACITIES[0] + CAPACITIES[1]) {
            dv.push(());
            assert_eq!(dv[i], ());
            assert_eq!(unsafe { *dv.get_unchecked(i) }, ());
        }
    }

    #[test]
    fn index_mut() {
        let mut dv = DycoVec::new();

        for i in 0..CAPACITIES[0] {
            dv.push(());
            assert_eq!(dv[i], ());
            dv[i] = ();
            assert_eq!(dv[i], ());
        }

        for i in 0..CAPACITIES[1] {
            dv.push(());
            assert_eq!(dv[i + CAPACITIES[0]], ());
            dv[i + CAPACITIES[0]] = ();
            assert_eq!(dv[i + CAPACITIES[0]], ());
        }
    }

    #[test]
    fn from_slice() {
        let arr = [1; CAPACITIES[0] + CAPACITIES[1]];
        let dv = DycoVec::from_boxed_slice(Box::new(arr));
        assert_eq!(dv.into_vec(), arr.to_vec());
    }

    #[test]
    fn from_slice_zst() {
        let arr = [(); CAPACITIES[0] + CAPACITIES[1]];
        let dv = DycoVec::from_boxed_slice(Box::new(arr));
        assert_eq!(dv.into_vec(), arr.to_vec());
    }

    #[test]
    fn find() {
        let dv = DycoVec::new();

        for i in 0..1000 {
            dv.push(i);
        }

        let index = dv.push(-1);

        for i in 1000..2000 {
            dv.push(i);
        }

        let index2 = dv.push(-2);

        assert_eq!(Some(index),
            dv
                .iter()
                .enumerate()
                .find_map(|(k, &v)| if v == -1 {
                    Some(k)
                } else {
                    None
                })
        );
        assert_eq!(Some(index2),
            dv
                .iter()
                .enumerate()
                .find_map(|(k, &v)| if v == -2 {
                    Some(k)
                } else {
                    None
                })
        );
    }

    #[test]
    fn max_id() {
        assert_eq!(MAX_ID + 1, CAPACITIES.iter().sum())
    }
}
