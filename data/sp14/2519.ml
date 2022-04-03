
let rec wwhile (f,b) =
  let rec wwhelper f b =
    let (b',c') = f b in if c' = false then b' else wwhelper f b' in
  wwhelper f b;;

let fixpoint (f,b) =
  wwhile ((let g x = let xx = f x in (xx, (xx != b)) in g b), b);;


(* fix

let rec wwhile (f,b) =
  let rec wwhelper f b =
    let (b',c') = f b in if c' = false then b' else wwhelper f b' in
  wwhelper f b;;

let fixpoint (f,b) =
  wwhile ((let g x = let xx = f x in (b, (xx != b)) in g), b);;

*)

(* changed spans
(8,38)-(8,40)
(8,56)-(8,59)
(8,62)-(8,63)
*)

(* type error slice
(2,3)-(5,16)
(2,16)-(5,14)
(4,18)-(4,19)
(4,18)-(4,21)
(4,52)-(4,60)
(4,52)-(4,65)
(4,61)-(4,62)
(5,2)-(5,10)
(5,2)-(5,14)
(5,11)-(5,12)
(8,2)-(8,8)
(8,2)-(8,64)
(8,9)-(8,64)
(8,10)-(8,60)
(8,17)-(8,52)
(8,21)-(8,52)
(8,37)-(8,52)
(8,56)-(8,57)
(8,56)-(8,59)
*)

(* all spans
(2,16)-(5,14)
(3,2)-(5,14)
(3,19)-(4,65)
(3,21)-(4,65)
(4,4)-(4,65)
(4,18)-(4,21)
(4,18)-(4,19)
(4,20)-(4,21)
(4,25)-(4,65)
(4,28)-(4,38)
(4,28)-(4,30)
(4,33)-(4,38)
(4,44)-(4,46)
(4,52)-(4,65)
(4,52)-(4,60)
(4,61)-(4,62)
(4,63)-(4,65)
(5,2)-(5,14)
(5,2)-(5,10)
(5,11)-(5,12)
(5,13)-(5,14)
(7,14)-(8,64)
(8,2)-(8,64)
(8,2)-(8,8)
(8,9)-(8,64)
(8,10)-(8,60)
(8,17)-(8,52)
(8,21)-(8,52)
(8,30)-(8,33)
(8,30)-(8,31)
(8,32)-(8,33)
(8,37)-(8,52)
(8,38)-(8,40)
(8,42)-(8,51)
(8,43)-(8,45)
(8,49)-(8,50)
(8,56)-(8,59)
(8,56)-(8,57)
(8,58)-(8,59)
(8,62)-(8,63)
*)