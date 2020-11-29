use core::{
    ptr::NonNull,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    convert::{AsRef, AsMut, TryFrom},
    fmt::{Debug, Formatter, Result as FmtResult},
    sync::atomic::{AtomicBool, AtomicUsize, AtomicPtr, Ordering, spin_loop_hint},
};
use alloc::{
    vec::Vec,
    boxed::Box,
    alloc::{alloc, dealloc, Layout},
};

pub(crate) struct Segment<T> {
    /// A pointer to the beginning of the contiguous array.
    ///
    /// The only time this is written to is when it's going from null to
    /// a pointer to the contiguous memory.
    pub head: AtomicPtr<T>,
    /// The capacity of the [`Segment`] *in terms of `T`*.
    pub cap: usize,
    /// The length of the [`Segment`] *in terms of `T`*.
    len: AtomicUsize,
    /// A boolean lock to prevent `push` allocation during `clone_from` or
    /// multiple `push` allocations.
    cas_lock: AtomicBool,
}

impl<T> AsRef<[T]> for Segment<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T> AsMut<[T]> for Segment<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T: Clone> Clone for Segment<T> {
    fn clone(&self) -> Self {
        let mut seg = Segment {
            head: AtomicPtr::new(core::ptr::null_mut()),
            cap: 0,
            len: AtomicUsize::new(0),
            cas_lock: AtomicBool::new(false),
        };

        seg.clone_from(self);
        seg
    }

    fn clone_from(&mut self, source: &Self) {
        source.lock();
        
        self.cap = source.cap;
        let len = source.len.load(Ordering::Acquire);
        self.len = AtomicUsize::new(len);

        let src = source.head.load(Ordering::Acquire);

        if src.is_null() {
            let lyt = Layout::array::<T>(self.cap).unwrap();
            let dst = unsafe { alloc(lyt) } as *mut T;
            unsafe {
                (&mut*core::ptr::slice_from_raw_parts_mut(dst, self.cap)).clone_from_slice(
                    &*core::ptr::slice_from_raw_parts(src, self.cap)
                );
            }

            self.head.store(dst, Ordering::Release);
        } else {
            self.head.store(src, Ordering::Release);
        }

        source.unlock();
    }
}

impl<T> Drop for Segment<T> {
    fn drop(&mut self) {
        if self.head.get_mut().is_null() {
            let lyt = Layout::array::<T>(self.cap).unwrap();
            unsafe { dealloc(*self.head.get_mut() as _, lyt) };
        }
    }
}

impl<T> Deref for Segment<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for Segment<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: safe, as the exclusive reference 
        self.as_slice_mut()
    }
}

impl<T: PartialEq> PartialEq for Segment<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cap == other.cap && self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for Segment<T> {}

impl<T: PartialEq> PartialEq<[T]> for Segment<T> {
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == other
    }
}

impl<T: PartialEq> PartialEq<Segment<T>> for [T] {
    fn eq(&self, other: &Segment<T>) -> bool {
        other == self
    }
}

impl<T: PartialEq> PartialEq<&'_ [T]> for Segment<T> {
    fn eq(&self, other: &&[T]) -> bool {
        self == *other
    }
}

impl<T: PartialEq> PartialEq<&'_ mut [T]> for Segment<T> {
    fn eq(&self, other: &&mut [T]) -> bool {
        self == *other
    }
}

impl<T: PartialEq> PartialEq<Segment<T>> for &'_ [T] {
    fn eq(&self, other: &Segment<T>) -> bool {
        other == self
    }
}

impl<T: PartialEq> PartialEq<Segment<T>> for &'_ mut [T] {
    fn eq(&self, other: &Segment<T>) -> bool {
        other == self
    }
}

impl<T> IntoIterator for Segment<T> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        Vec::into_iter(self.into())
    }
}

impl<T> From<Segment<T>> for Vec<T> {
    fn from(s: Segment<T>) -> Self {
        s.into_boxed_slice().into_vec()
    }
}

impl<T: Hash> Hash for Segment<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        T::hash_slice(self, state)
    }
}

impl<T: Debug> Debug for Segment<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        self.as_slice().fmt(f)
    }
}

impl<T: PartialOrd> PartialOrd for Segment<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other)
    }
}

impl<T: Ord> Ord for Segment<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_slice().cmp(other)
    }
}

impl<T> Segment<T> {
    const T_SIZE: usize = core::mem::size_of::<T>();
    const T_ZST: bool = Self::T_SIZE == 0;

    pub const fn new(head: *mut T, cap: usize) -> Self {
        Segment {
            head: AtomicPtr::new(head),
            cap,
            len: AtomicUsize::new(0),
            cas_lock: AtomicBool::new(false),
        }
    }

    pub const fn new_null(cap: usize) -> Self {
        Self::new(core::ptr::null_mut(), cap)
    }

    #[allow(clippy::mut_mut)]
    pub fn from_slice(s: &mut &mut [T], max_cap: usize) -> Self {
        let range = s.len().min(max_cap + 1);
        let ptr = unsafe { alloc(Layout::array::<T>(max_cap).unwrap()) } as *mut T;
        let sp = s.as_ptr();

        unsafe { core::ptr::copy_nonoverlapping(sp, ptr, range); }
        *s = &mut [];
        let mut out = Self::new(ptr, max_cap);
        *out.len.get_mut() = range;
        out
    }

    pub fn as_slice(&self) -> &[T] {
        if Self::T_ZST {
            unsafe { core::slice::from_raw_parts(NonNull::dangling().as_ptr(), self.len()) }
        } else {
            let head = self.head.load(Ordering::Acquire);

            if head.is_null() {
                &[]
            } else {
                let len = self.len();
                debug_assert!(isize::try_from(len.saturating_mul(Self::T_SIZE)).is_ok());
                /* SAFE:
                * The data is properly aligned, as allocated by `alloc`
                    * one contiguous allocation
                    * `ptr` is NonNull
                * the data does point to the correct number of consecutive,
                initialized elements
                * the data cannot be mutated
                * length validity is debug-checked above
                */
                unsafe { core::slice::from_raw_parts(head, len) }
            }
        }
    }

    /// Gets a mutable slice from `self`.
    #[allow(clippy::mut_mut)]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if Self::T_ZST {
            unsafe { core::slice::from_raw_parts_mut(NonNull::dangling().as_ptr(), self.len()) }
        } else {
            let head = self.head.load(Ordering::Acquire);

            if head.is_null() {
                &mut []
            } else {
                let len = self.len();
                debug_assert!(isize::try_from(len.saturating_mul(Self::T_SIZE)).is_ok());

                /* SAFE:
                * The data is properly aligned, as allocated by `alloc`
                    * one contiguous allocation
                    * `ptr` is NonNull
                * the data does point to the correct number of consecutive,
                initialized elements
                * can't access or read through self (this is an exclusive borrow of self)
                * length validity is debug-checked above
                */
                unsafe { core::slice::from_raw_parts_mut(head, len) }
            }
        }
    }

    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        let b = unsafe { Box::<[T]>::from_raw(self.as_slice_mut()) };
        core::mem::forget(self);
        b
    }

    pub fn clear(&mut self) {
        if self.head.get_mut().is_null() {
            for i in 0..*self.len.get_mut() {
                /* SAFE:
                    * the pointer is valid for reads and writes
                    * the pointer is properly aligned
                */
                unsafe { core::ptr::drop_in_place(self.head.get_mut().add(i)) };
            }
        }

        self.len = AtomicUsize::new(0);
    }

    #[cold]
    fn push_slow_path(&self) -> *mut T {
        self.lock();
        
        let head = if self.is_initialized() {
            self.head.load(Ordering::Acquire)
        } else {
            let head = unsafe { alloc(Layout::array::<T>(self.cap).unwrap()) } as *mut T;
            self.head.store(head, Ordering::Release);
            head
        };
        
        self.unlock();
        head
    }

    #[must_use]
    pub fn push(&self, t: &mut *const T) -> Option<usize> {
        if Self::T_ZST {
            let offset = self.len.fetch_add(1, Ordering::AcqRel);

            if offset < self.cap {
                Some(offset)
            } else {
                None
            }
        } else {
            let offset = self.len.fetch_add(1, Ordering::AcqRel);
            let mut head = self.head.load(Ordering::Acquire);

            if offset >= self.cap {
                return None;
            } else if head.is_null() {
                head = self.push_slow_path();
            };

            unsafe {
                head = head.add(offset);
                *head = core::mem::replace(t, core::ptr::null()).read();
            }

            Some(offset)
        }
    }

    pub fn is_initialized(&self) -> bool {
        Self::T_ZST || !self.head.load(Ordering::Acquire).is_null()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) {
        while self.cas_lock
            .compare_exchange_weak(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        { spin_loop_hint() };
    }

    fn unlock(&self) {
        self.cas_lock.store(false, Ordering::Release);
    }

    fn len(&self) -> usize {
        let len = self.len.load(Ordering::Acquire);
        debug_assert!(len <= self.cap + 1);
        len.min(self.cap)
    }
}

#[cfg(test)]
mod test {
    #[allow(unused_imports)]
    use super::*;

    fn my_alloc<T>(cap: usize) -> *mut T {
        unsafe { alloc(Layout::array::<T>(cap).unwrap()) as *mut T }
    }

    #[test]
    fn new_null_is_new() {
        let cap = 0;
        let ptr = core::ptr::null_mut::<i32>();
        assert_eq!(Segment::new_null(cap), Segment::new(ptr, cap));
    }

    #[test]
    fn nonnull_is_init() {
        let ptr = my_alloc::<i32>(1);
        assert!(Segment::new(ptr, 0).is_initialized());
    }

    #[test]
    fn null_nonzst_is_uninit() {
        assert!(!Segment::<i32>::new_null(0).is_initialized());
    }

    #[test]
    fn null_zst_is_init() {
        assert!(Segment::<()>::new_null(0).is_initialized());
    }

    #[test]
    fn push_is_init() {
        let seg = Segment::new_null(1);
        assert!(!seg.is_initialized());
        assert!(seg.push(&mut (&true as *const _)).is_some());
        assert!(seg.is_initialized());
    }

    #[test]
    fn len() {
        let seg = Segment::new_null(5);
        assert_eq!(seg.len(), 0);
        assert!(seg.push(&mut (&1 as *const _)).is_some());
        assert!(seg.push(&mut (&2 as *const _)).is_some());
        assert!(seg.push(&mut (&3 as *const _)).is_some());
        assert!(seg.push(&mut (&4 as *const _)).is_some());
        assert!(seg.push(&mut (&5 as *const _)).is_some());
        assert_eq!(seg.len(), 5);
        assert!(seg.push(&mut (&6 as *const _)).is_none());
        assert_eq!(seg.len(), 5);
    }

    #[test]
    fn len_zst() {
        let seg = Segment::new_null(5);
        assert_eq!(seg.len(), 0);
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert_eq!(seg.len(), 5);
        assert!(seg.push(&mut (&() as *const _)).is_none());
        assert_eq!(seg.len(), 5);
    }

    #[test]
    fn as_slice() {
        let seg = Segment::new_null(3);
        assert!(seg.push(&mut (&1 as *const _)).is_some());
        assert!(seg.push(&mut (&2 as *const _)).is_some());
        assert!(seg.push(&mut (&3 as *const _)).is_some());
        assert_eq!(seg.as_slice(), &[1, 2, 3]);
        assert!(seg.push(&mut (&4 as *const _)).is_none());
        assert_eq!(seg.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn as_slice_zst() {
        let seg = Segment::new_null(3);
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert_eq!(seg.as_slice(), &[(), (), ()]);
        assert!(seg.push(&mut (&() as *const _)).is_none());
        assert_eq!(seg.as_slice(), &[(), (), ()]);
    }

    #[test]
    fn clear() {
        let mut seg = Segment::new_null(3);
        assert!(seg.push(&mut (&1 as *const _)).is_some());
        assert!(seg.push(&mut (&2 as *const _)).is_some());
        assert!(seg.push(&mut (&3 as *const _)).is_some());
        assert!(seg.push(&mut (&4 as *const _)).is_none());
        seg.clear();
        assert_eq!(seg.len(), 0);
        assert!(seg.push(&mut (&1 as *const _)).is_some());
        assert!(seg.push(&mut (&2 as *const _)).is_some());
        assert!(seg.push(&mut (&3 as *const _)).is_some());
        assert!(seg.push(&mut (&4 as *const _)).is_none());
    }

    #[test]
    fn clear_zst() {
        let mut seg = Segment::new_null(3);
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_none());
        seg.clear();
        assert_eq!(seg.len(), 0);
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_none());
    }

    #[test]
    fn index() {
        let seg = Segment::new_null(3);
        assert!(seg.push(&mut (&1 as *const _)).is_some());
        assert!(seg.push(&mut (&2 as *const _)).is_some());
        assert!(seg.push(&mut (&3 as *const _)).is_some());
        assert_eq!(seg[0], 1);
        assert_eq!(seg[1], 2);
        assert_eq!(seg[2], 3);
    }

    #[test]
    fn index_zst() {
        let seg = Segment::new_null(3);
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert_eq!(seg[0], ());
        assert_eq!(seg[1], ());
        assert_eq!(seg[2], ());
    }

    #[test]
    fn index_mut() {
        let mut seg = Segment::new_null(3);
        assert!(seg.push(&mut (&1 as *const _)).is_some());
        assert!(seg.push(&mut (&2 as *const _)).is_some());
        assert!(seg.push(&mut (&3 as *const _)).is_some());
        seg[0] = 3;
        seg[1] = 2;
        seg[2] = 1;
        assert_eq!(seg[0], 3);
        assert_eq!(seg[1], 2);
        assert_eq!(seg[2], 1);
    }

    #[test]
    fn index_mut_zst() {
        let mut seg = Segment::new_null(3);
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        assert!(seg.push(&mut (&() as *const _)).is_some());
        seg[0] = ();
        seg[1] = ();
        seg[2] = ();
        assert_eq!(seg[0], ());
        assert_eq!(seg[1], ());
        assert_eq!(seg[2], ());
    }
}
