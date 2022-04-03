
let rec app l1 l2 = match l1 with | [] -> l2 | h::t -> h :: (app t l2);;

let rec digitsOfInt n =
  if n >= 10 then app (digitsOfInt (n / 10) [n mod 10]) else app [n] [8];;


(* fix

let rec app l1 l2 = match l1 with | [] -> l2 | h::t -> h :: (app t l2);;

let rec digitsOfInt n = if n >= 10 then app [5] [n mod 10] else app [3] [8];;

*)

(* changed spans
(5,18)-(5,21)
(5,18)-(5,55)
(5,23)-(5,34)
(5,35)-(5,43)
(5,36)-(5,37)
(5,40)-(5,42)
(5,45)-(5,53)
(5,66)-(5,67)
*)

(* type error slice
(5,2)-(5,72)
(5,18)-(5,21)
(5,18)-(5,55)
(5,61)-(5,64)
(5,61)-(5,72)
*)

(* all spans
(2,12)-(2,70)
(2,15)-(2,70)
(2,20)-(2,70)
(2,26)-(2,28)
(2,42)-(2,44)
(2,55)-(2,70)
(2,55)-(2,56)
(2,60)-(2,70)
(2,61)-(2,64)
(2,65)-(2,66)
(2,67)-(2,69)
(4,20)-(5,72)
(5,2)-(5,72)
(5,5)-(5,12)
(5,5)-(5,6)
(5,10)-(5,12)
(5,18)-(5,55)
(5,18)-(5,21)
(5,22)-(5,55)
(5,23)-(5,34)
(5,35)-(5,43)
(5,36)-(5,37)
(5,40)-(5,42)
(5,44)-(5,54)
(5,45)-(5,53)
(5,45)-(5,46)
(5,51)-(5,53)
(5,61)-(5,72)
(5,61)-(5,64)
(5,65)-(5,68)
(5,66)-(5,67)
(5,69)-(5,72)
(5,70)-(5,71)
*)