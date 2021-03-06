
let rec wwhile (f,b) =
  match f b with | (a,b) -> if not b then a else wwhile (f, a);;

let fixpoint (f,b) = wwhile ((f (if (f b) = b then b else b)), b);;


(* fix

let rec wwhile (f,b) =
  match f b with | (a,b) -> if not b then a else wwhile (f, a);;

let fixpoint (f,b) =
  wwhile ((let f x = let xx = (x * x) * x in (xx, (xx < 100)) in f), b);;

*)

(* changed spans
(5,29)-(5,61)
(5,30)-(5,31)
(5,32)-(5,60)
(5,36)-(5,41)
(5,36)-(5,45)
(5,37)-(5,38)
(5,44)-(5,45)
(5,51)-(5,52)
(5,58)-(5,59)
(5,63)-(5,64)
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
(5,21)-(5,27)
(5,21)-(5,65)
(5,28)-(5,65)
(5,29)-(5,61)
(5,30)-(5,31)
(5,36)-(5,41)
(5,36)-(5,45)
(5,37)-(5,38)
(5,44)-(5,45)
(5,63)-(5,64)
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
(5,14)-(5,65)
(5,21)-(5,65)
(5,21)-(5,27)
(5,28)-(5,65)
(5,29)-(5,61)
(5,30)-(5,31)
(5,32)-(5,60)
(5,36)-(5,45)
(5,36)-(5,41)
(5,37)-(5,38)
(5,39)-(5,40)
(5,44)-(5,45)
(5,51)-(5,52)
(5,58)-(5,59)
(5,63)-(5,64)
*)
