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

mod vs_2d {
    use super::*;

    fn evmap_push_only<T: Eq + Hash + ShallowCopy + Copy + Send + Sync>(
        n: usize,
        m: usize,
        t: T,
    ) -> (ReadHandle<usize, T>, Mutex<WriteHandle<usize, T>>) {
        let (read, write) = evmap::new();
        let write = Mutex::new(write);
    
        (0..n)
            .into_par_iter()
            .for_each(|i| for _ in 0..m {
                let mut write = write.lock();
                write.insert(i, t);
                write.refresh();
            });

        (read, write)
    }

    fn dv_push_only<T: Copy + Send + Sync>(n: usize, m: usize, t: T) -> DycoVec<DycoVec<T>> {
        let dv = DycoVec::new();

        (0..n)
            .into_par_iter()
            .for_each(|_| {
                let i = dv.push(DycoVec::new());

                for _ in 0..m {
                    dv[i].push(t);
                }
            });

        dv
    }

    fn segqueue_push_only<T: Copy + Send + Sync>(
        n: usize,
        m: usize,
        t: T
    ) -> SegQueue<SegQueue<T>> {
        let queue = SegQueue::new();

        (0..n)
            .into_par_iter()
            .for_each(|_| {
                let new = SegQueue::new();

                for _ in 0..m {
                    new.push(t);
                }

                queue.push(new);
            });

        queue
    }

    fn arrayqueue_push_only<T: Copy + Debug + Send + Sync>(
        n: usize,
        m: usize,
        t: T,
    ) -> ArrayQueue<ArrayQueue<T>> {
        let queue = ArrayQueue::new(n);

        (0..n)
            .into_par_iter()
            .for_each(|_| {
                let new = ArrayQueue::new(m);

                for _ in 0..m {
                    new.push(t).unwrap();
                }

                queue.push(new).unwrap();
            });

        queue
    }

    fn slab_push_only<T: Copy + Send + Sync>(n: usize, m: usize, t: T) -> Slab<Slab<T>> {
        let slab = Slab::new();

        (0..n)
            .into_par_iter()
            .for_each(|_| {
                let new = Slab::new();

                for _ in 0..m {
                    new.insert(t).unwrap();
                }

                slab.insert(new).unwrap();
            });

        slab
    }

    fn vec_push_only<T: Copy + Send + Sync>(n: usize, m: usize, t: T) -> Mutex<Vec<Vec<T>>> {
        let vec = Mutex::new(Vec::new());

        (0..n)
            .into_par_iter()
            .for_each(|_| {
                let mut new = Vec::new();

                for _ in 0..m {
                    new.push(t);
                }

                vec.lock().push(new);
            });

        vec
    }

    pub fn bench_2d_small<M: Measurement>(
        c: &mut criterion::BenchmarkGroup<'_, M>,
        n: usize,
        m: usize
    ) {
        let v = SmallData::default();

        c.bench_function("evmap push only, Small", |b|
            b.iter(|| evmap_push_only(n, m, black_box(v)))
        );
        c.bench_function("DycoVec push only, Small", |b|
            b.iter(|| dv_push_only(n, m, black_box(v)))
        );
        c.bench_function("SegQueue push only, Small", |b|
            b.iter(|| segqueue_push_only(n, m, black_box(v)))
        );
        c.bench_function("ArrayQueue push only, Small", |b|
            b.iter(|| arrayqueue_push_only(n, m, black_box(v)))
        );
        c.bench_function("Slab push only, Small", |b|
            b.iter(|| slab_push_only(n, m, black_box(v)))
        );
        c.bench_function("Mutex<Vec> push only, Small", |b|
            b.iter(|| vec_push_only(n, m, black_box(v)))
        );
    }

    pub fn bench_2d_big<M: Measurement>(
        c: &mut criterion::BenchmarkGroup<'_, M>,
        n: usize,
        m: usize
    ) {
        let v = BigData::default();

        c.bench_function("evmap push only, Big", |b|
            b.iter(|| evmap_push_only(n, m, black_box(v)))
        );
        c.bench_function("DycoVec push only, Big", |b|
            b.iter(|| dv_push_only(n, m, black_box(v)))
        );
        c.bench_function("SegQueue push only, Big", |b|
            b.iter(|| segqueue_push_only(n, m, black_box(v)))
        );
        c.bench_function("ArrayQueue push only, Big", |b|
            b.iter(|| arrayqueue_push_only(n, m, black_box(v)))
        );
        c.bench_function("Slab push only, Big", |b|
            b.iter(|| slab_push_only(n, m, black_box(v)))
        );
        c.bench_function("Mutex<Vec> push only, Big", |b|
            b.iter(|| vec_push_only(n, m, black_box(v)))
        );
    }
}

fn rectangle_push_only(c: &mut Criterion) {
    {
        let mut po_75x75 = c.benchmark_group("Par 2D, 75 x 75, Push Only");
        vs_2d::bench_2d_small(&mut po_75x75, 75, 75);
        vs_2d::bench_2d_big(&mut po_75x75, 75, 75);
    }

    {
        let mut po_100x100 = c.benchmark_group("Par 2D, 100 x 100, Push Only");
        vs_2d::bench_2d_small(&mut po_100x100, 100, 100);
        vs_2d::bench_2d_big(&mut po_100x100, 100, 100);
    }

    {
        let mut po_200x50 = c.benchmark_group("Par 2D, 200 x 50, Push Only");
        vs_2d::bench_2d_small(&mut po_200x50, 200, 50);
        vs_2d::bench_2d_big(&mut po_200x50, 200, 50);
    }

    {
        let mut po_50x200 = c.benchmark_group("Par 2D, 50 x 200, Push Only");
        vs_2d::bench_2d_small(&mut po_50x200, 50, 200);
        vs_2d::bench_2d_big(&mut po_50x200, 50, 200);
    }
}

criterion_group!(par_rectangles, rectangle_push_only);
criterion_main!(par_rectangles);
