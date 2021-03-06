
let rec wwhile (f,b) =
  match f b with | (a,b) -> if not b then a else wwhile (f, a);;

let fixpoint (f,b) = wwhile ((fun x  -> x), b);;


(* fix

let rec wwhile (f,b) =
  match f b with | (a,b) -> if not b then a else wwhile (f, a);;

let fixpoint (f,b) =
  wwhile ((let f x = let xx = (x * x) * x in (xx, (xx < 100)) in f), b);;

*)

(* changed spans
(5,29)-(5,42)
(5,40)-(5,41)
(5,44)-(5,45)
*)

(* type error slice
(2,3)-(3,64)
(2,16)-(3,62)
(3,2)-(3,62)
(3,8)-(3,9)
(3,8)-(3,11)
(3,10)-(3,11)
(3,49)-(3,55)
(3,49)-(3,62)
(3,56)-(3,62)
(3,60)-(3,61)
(5,21)-(5,27)
(5,21)-(5,46)
(5,28)-(5,46)
(5,29)-(5,42)
(5,40)-(5,41)
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
(5,14)-(5,46)
(5,21)-(5,46)
(5,21)-(5,27)
(5,28)-(5,46)
(5,29)-(5,42)
(5,40)-(5,41)
(5,44)-(5,45)
*)
