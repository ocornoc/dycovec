use std::mem::ManuallyDrop;
use std::hash::Hash;
use std::fmt::Debug;
use dycovec::DycoVec;
use evmap::{ReadHandle, WriteHandle, ShallowCopy};
use evmap_derive::*;
use crossbeam::queue::{SegQueue, ArrayQueue};
use sharded_slab::Slab;
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

    fn evmap_push_only<T: Eq + Hash + ShallowCopy + Copy>(
        n: usize,
        t: T,
    ) -> (ReadHandle<usize, T>, WriteHandle<usize, T>) {
        let (read, mut write) = evmap::new();
    
        for _ in 0..n {
            write.insert(0, t);
        }

        write.refresh();
        (read, write)
    }

    fn dv_push_only<T: Copy>(n: usize, t: T) -> DycoVec<T> {
        let dv = DycoVec::new();

        for _ in 0..n {
            dv.push(t);
        }

        dv
    }

    fn segqueue_push_only<T: Copy>(n: usize, t: T) -> SegQueue<T> {
        let queue = SegQueue::new();

        for _ in 0..n {
            queue.push(t);
        }

        queue
    }

    fn arrayqueue_push_only<T: Copy + Debug>(n: usize, t: T) -> ArrayQueue<T> {
        let queue = ArrayQueue::new(n);

        for _ in 0..n {
            queue.push(t).unwrap();
        }

        queue
    }

    fn slab_push_only<T: Copy>(n: usize, t: T) -> Slab<T> {
        let slab = Slab::new();

        for _ in 0..n {
            slab.insert(t).unwrap();
        }

        slab
    }

    fn vec_push_only<T: Copy>(n: usize, t: T) -> Vec<T> {
        let mut vec = Vec::new();

        for _ in 0..n {
            vec.push(t);
        }

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
        c.bench_function("Vec push only, Small", |b|
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
        c.bench_function("Vec push only, Big", |b|
            b.iter(|| vec_push_only(n, black_box(v)))
        );
    }
}

fn line_push_only(c: &mut Criterion) {
    {
        let mut po = c.benchmark_group("Seq 1D, 100, Push Only");
        vs_1d::bench_1d_small(&mut po, 100);
        vs_1d::bench_1d_big(&mut po, 100);
    }

    {
        let mut po = c.benchmark_group("Seq 1D, 1000, Push Only");
        vs_1d::bench_1d_small(&mut po, 1000);
        vs_1d::bench_1d_big(&mut po, 1000);
    }

    {
        let mut po = c.benchmark_group("Seq 1D, 5000, Push Only");
        vs_1d::bench_1d_small(&mut po, 5000);
        vs_1d::bench_1d_big(&mut po, 5000);
    }

    {
        let mut po = c.benchmark_group("Seq 1D, 10000, Push Only");
        vs_1d::bench_1d_small(&mut po, 10000);
        vs_1d::bench_1d_big(&mut po, 10000);
    }
}

criterion_group!(seq_line, line_push_only);
criterion_main!(seq_line);
