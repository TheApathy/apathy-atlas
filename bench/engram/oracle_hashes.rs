use std::convert::TryInto;
fn rd_i32(p:&str)->Vec<i32>{let b=std::fs::read(p).unwrap();b.chunks(4).map(|c|i32::from_le_bytes(c.try_into().unwrap())).collect()}
fn main(){
    let a:Vec<String>=std::env::args().collect();
    let mode = a.get(1).map(|s|s.as_str()).unwrap_or("good");
    let ids:Vec<u32>=vec![3465,10150,3913,3395,361,260,14,291,438,223,18,14,223,19,201,361,362,944,295,
                          3291,3913,3395,528,260,14,291,438,291,14,260,940,291,201,361,1354,260,201];
    let mut token_map = rd_i32("token_map_i32.bin");
    let mut mult = vec![
        vec![76632096046245i64, 4839876093313, 35959672319349, 73987337458391],
        vec![67716810739261i64, 51510806800915, 30921347202721, 82619226485591]];
    let mut primes_swapped = false;
    match mode {
        "good" => {}
        // NEG 1: flip one multiplier's low bit (stays odd by +2)
        "neg-mult" => { mult[0][2] += 2; }
        // NEG 2: permute the prime order
        "neg-primes" => { primes_swapped = true; }
        // NEG 3 (team lead's): perturb token_map at ONE id. This failure would
        // otherwise be mistaken for a tokenizer-version difference, not a bug.
        "neg-tokenmap" => { let t = ids[5] as usize; token_map[t] = (token_map[t] + 1) % 99092; }
        _ => panic!("unknown mode"),
    }
    let mut layout = EngramLayout::new(&[1,14], 4, 8, 256, 16_000_000).unwrap();
    if primes_swapped {
        let x = layout.primes[0][0][0]; let y = layout.primes[0][0][1];
        layout.primes[0][0][0] = y; layout.primes[0][0][1] = x;
    } else {
        layout.validate_against_config(&[384_006_168, 384_016_682]).unwrap();
    }
    let mut st = EngramHashState::new(layout, token_map, mult, 2, 99_092).unwrap();
    let out = st.forward(&ids, 0, None).unwrap();
    let t = ids.len();
    for (li,name) in [(0usize,"L01"),(1usize,"L14")] {
        let mut b=Vec::with_capacity(t*24*8);
        for tok in 0..t { for c in 0..24 { b.extend_from_slice(&out[(tok*2+li)*24+c].to_le_bytes()); } }
        std::fs::write(format!("cand_{name}_{mode}.bin"), b).unwrap();
    }
    println!("{mode}: wrote cand_L01_{mode}.bin cand_L14_{mode}.bin  ({t} tokens x 24)");
}
