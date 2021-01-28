# `DycoVec`: A **dy**namically-allocated, **co**ncurrent **vec**tor.

`DycoVec`s are thread-safe vectors that support random access, iteration,
and tail pushing in an `O(1)`, wait-free\* manner, *and that's it*! For
the sake of performance, there is no support for *concurrent* clearing or
deletion, though this still can be performed given a mutable reference.

```rust
use dycovec::DycoVec;

fn main() {
    let dv = vec![1, 2, 3, 4].into_iter().collect::<DycoVec<i32>>();
    let t1 = std::thread::spawn(|| {
        dv.push(10);
    });
    let t2 = std::thread::spawn(|| {
        dv.push(20);
    });

    t1.join().unwrap();
    t2.join().unwrap();

    println!("{:?}", dv);
}
```

## \*Almost Wait-free

The `DycoVec` is almost always wait-free to push on to, and is otherwise
still lock-free. The only exception to wait-freedom is when a segment has
filled and a new one is being allocated. This will result in every thread
simultaneously attempting to enter a specific critical section. The first
to do so will allocate the segment. After finishing, all other threads
vying for the lock will enter the critical section as appropriate, notice
the segment is already allocated, and immediately unlock. So, the maximum
hold-up will be O(`n`) (`n` being the number of threads). But, as a
reminder, this whole parade only happens when the `DycoVec` has yet to
be allocated.

The only other time the `DycoVec` is locked is during cloning as to prevent
an incomplete allocation from providing bad data for the clone.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
