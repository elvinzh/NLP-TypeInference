
let rec wwhile (f,b) =
  match f b with | (a,b) -> if not b then a else wwhile (f, a);;

let fixpoint (f,b) =
  let g x = let xx = f x in (xx, (xx != b)) in wwhile (f, b);;


(* fix

let rec wwhile (f,b) =
  match f b with | (a,b) -> if not b then a else wwhile (f, a);;

let fixpoint (f,b) =
  let g x = let xx = f x in (xx, (xx != x)) in wwhile (g, b);;

*)

(* changed spans
(6,40)-(6,41)
(6,55)-(6,56)
*)

(* type error slice
(3,2)-(3,62)
(3,8)-(3,9)
(3,8)-(3,11)
(3,49)-(3,55)
(3,49)-(3,62)
(3,56)-(3,62)
(3,57)-(3,58)
(3,60)-(3,61)
(6,12)-(6,43)
(6,21)-(6,22)
(6,21)-(6,24)
(6,33)-(6,42)
(6,34)-(6,36)
(6,40)-(6,41)
(6,47)-(6,53)
(6,47)-(6,60)
(6,54)-(6,60)
(6,55)-(6,56)
(6,58)-(6,59)
*)

(* all spans
(2,16)-(3,62)
(3,2)-(3,62)
(3,8)-(3,11)
(3,8)-(3,9)
(3,10)-(3,11)
(3,28)-(3,62)
(3,31)-(3,36)
(3,31)-(3,34)
(3,35)-(3,36)
(3,42)-(3,43)
(3,49)-(3,62)
(3,49)-(3,55)
(3,56)-(3,62)
(3,57)-(3,58)
(3,60)-(3,61)
(5,14)-(6,60)
(6,2)-(6,60)
(6,8)-(6,43)
(6,12)-(6,43)
(6,21)-(6,24)
(6,21)-(6,22)
(6,23)-(6,24)
(6,28)-(6,43)
(6,29)-(6,31)
(6,33)-(6,42)
(6,34)-(6,36)
(6,40)-(6,41)
(6,47)-(6,60)
(6,47)-(6,53)
(6,54)-(6,60)
(6,55)-(6,56)
(6,58)-(6,59)
*)
