use std::mem::ManuallyDrop;
use std::hash::Hash;
use std::fmt::Debug;
use rayon::prelude::*;
use dycovec::DycoVec;
use evmap::{ReadHandle, WriteHandle, ShallowCopy};
use evmap_derive::*;
use crossbeam::queue::{SegQueue, ArrayQueue};
use sharded_slab::Slab;
use parking_lot::Mutex;
use criterion::{black_box, criterion_group, criterion_main, Criterion, measurement::Measurement};

#[derive(Copy, Clone, Default, Eq, PartialEq, Hash, ShallowCopy, Debug)]
struct SmallData(usize);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct BigData([usize; 200]);

impl Default for BigData {
    fn default() -> Self {
        BigData([0; 200])
    }
}

impl ShallowCopy for BigData {
    unsafe fn shallow_copy(&self) -> ManuallyDrop<Self> {
        ManuallyDrop::new(*self)
    }
}

mod vs_1d {
    use super::*;

    fn evmap_push_only<T: Eq + Hash + ShallowCopy + Copy + Send + Sync>(
        n: usize,
        t: T,
    ) -> (ReadHandle<usize, T>, Mutex<WriteHandle<usize, T>>) {
        let (read, write) = evmap::new();
        let write = Mutex::new(write);
    
        (0..n).into_par_iter().for_each(|_| {
            let mut write = write.lock();
            write.insert(0, t);
        });

        write.lock().refresh();
        (read, write)
    }

    fn dv_push_only<T: Copy + Send + Sync>(n: usize, t: T) -> DycoVec<T> {
        let dv = DycoVec::new();

        (0..n).into_par_iter().for_each(|_| { dv.push(t); });

        dv
    }

    fn segqueue_push_only<T: Copy + Send + Sync>(n: usize, t: T) -> SegQueue<T> {
        let queue = SegQueue::new();

        (0..n).into_par_iter().for_each(|_| queue.push(t));

        queue
    }

    fn arrayqueue_push_only<T: Copy + Debug + Send + Sync>(n: usize, t: T) -> ArrayQueue<T> {
        let queue = ArrayQueue::new(n);

        (0..n).into_par_iter().for_each(|_| queue.push(t).unwrap());

        queue
    }

    fn slab_push_only<T: Copy + Send + Sync>(n: usize, t: T) -> Slab<T> {
        let slab = Slab::new();

        (0..n).into_par_iter().for_each(|_| { slab.insert(t).unwrap(); });

        slab
    }

    fn vec_push_only<T: Copy + Send + Sync>(n: usize, t: T) -> Mutex<Vec<T>> {
        let vec = Mutex::new(Vec::new());

        (0..n).into_par_iter().for_each(|_| vec.lock().push(t));

        vec
    }

    pub fn bench_1d_small<M: Measurement>(
        c: &mut criterion::BenchmarkGroup<'_, M>,
        n: usize,
    ) {
        let v = SmallData::default();

        c.bench_function("evmap push only, Small", |b|
            b.iter(|| evmap_push_only(n, black_box(v)))
        );
        c.bench_function("DycoVec push only, Small", |b|
            b.iter(|| dv_push_only(n, black_box(v)))
        );
        c.bench_function("SegQueue push only, Small", |b|
            b.iter(|| segqueue_push_only(n, black_box(v)))
        );
        c.bench_function("ArrayQueue push only, Small", |b|
            b.iter(|| arrayqueue_push_only(n, black_box(v)))
        );
        c.bench_function("Slab push only, Small", |b|
            b.iter(|| slab_push_only(n, black_box(v)))
        );
        c.bench_function("Mutex<Vec> push only, Small", |b|
            b.iter(|| vec_push_only(n, black_box(v)))
        );
    }

    pub fn bench_1d_big<M: Measurement>(
        c: &mut criterion::BenchmarkGroup<'_, M>,
        n: usize,
    ) {
        let v = BigData::default();

        c.bench_function("evmap push only, Big", |b|
            b.iter(|| evmap_push_only(n, black_box(v)))
        );
        c.bench_function("DycoVec push only, Big", |b|
            b.iter(|| dv_push_only(n, black_box(v)))
        );
        c.bench_function("SegQueue push only, Big", |b|
            b.iter(|| segqueue_push_only(n, black_box(v)))
        );
        c.bench_function("ArrayQueue push only, Big", |b|
            b.iter(|| arrayqueue_push_only(n, black_box(v)))
        );
        c.bench_function("Slab push only, Big", |b|
            b.iter(|| slab_push_only(n, black_box(v)))
        );
        c.bench_function("Mutex<Vec> push only, Big", |b|
            b.iter(|| vec_push_only(n, black_box(v)))
        );
    }
}

fn line_push_only(c: &mut Criterion) {
    {
        let mut po = c.benchmark_group("Par 1D, 100, Push Only");
        vs_1d::bench_1d_small(&mut po, 100);
        vs_1d::bench_1d_big(&mut po, 100);
    }

    {
        let mut po = c.benchmark_group("Par 1D, 1000, Push Only");
        vs_1d::bench_1d_small(&mut po, 1000);
        vs_1d::bench_1d_big(&mut po, 1000);
    }

    {
        let mut po = c.benchmark_group("Par 1D, 5000, Push Only");
        vs_1d::bench_1d_small(&mut po, 5000);
        vs_1d::bench_1d_big(&mut po, 5000);
    }

    {
        let mut po = c.benchmark_group("Par 1D, 10000, Push Only");
        vs_1d::bench_1d_small(&mut po, 10000);
        vs_1d::bench_1d_big(&mut po, 10000);
    }
}

criterion_group!(par_line, line_push_only);
criterion_main!(par_line);
