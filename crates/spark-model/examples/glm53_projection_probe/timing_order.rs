// SPDX-License-Identifier: AGPL-3.0-only
//! Five cyclic orders and their reverses, balancing each arm at each position.
pub fn order(index: usize) -> [usize; 5] {
    let index = index % 10;
    let mut order = std::array::from_fn(|position| (position + index % 5) % 5);
    if index >= 5 {
        order.reverse();
    }
    order
}
