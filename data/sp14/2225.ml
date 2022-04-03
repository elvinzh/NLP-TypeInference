
let rec wwhile (f,b) =
  match f b with | (b',true ) -> wwhile (f, b') | (b',false ) -> b';;

let fixpoint (f,b) = wwhile (((f b), (b <> (f b))), b);;


(* fix

let rec wwhile (f,b) =
  match f b with | (b',true ) -> wwhile (f, b') | (b',false ) -> b';;

let fixpoint (f,b) =
  let f b = let b' = f b in (b', ((f b) <> b)) in wwhile (f, b);;

*)

(* changed spans
(5,21)-(5,27)
(5,21)-(5,54)
(5,28)-(5,54)
(5,29)-(5,50)
(5,30)-(5,35)
(5,37)-(5,49)
(5,44)-(5,45)
(5,52)-(5,53)
*)

(* type error slice
(3,8)-(3,9)
(3,8)-(3,11)
(3,33)-(3,39)
(3,33)-(3,47)
(3,40)-(3,47)
(3,41)-(3,42)
(5,21)-(5,27)
(5,21)-(5,54)
(5,28)-(5,54)
(5,29)-(5,50)
*)

(* all spans
(2,16)-(3,67)
(3,2)-(3,67)
(3,8)-(3,11)
(3,8)-(3,9)
(3,10)-(3,11)
(3,33)-(3,47)
(3,33)-(3,39)
(3,40)-(3,47)
(3,41)-(3,42)
(3,44)-(3,46)
(3,65)-(3,67)
(5,14)-(5,54)
(5,21)-(5,54)
(5,21)-(5,27)
(5,28)-(5,54)
(5,29)-(5,50)
(5,30)-(5,35)
(5,31)-(5,32)
(5,33)-(5,34)
(5,37)-(5,49)
(5,38)-(5,39)
(5,43)-(5,48)
(5,44)-(5,45)
(5,46)-(5,47)
(5,52)-(5,53)
*)
