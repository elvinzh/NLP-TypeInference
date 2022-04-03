
let rec listReverse l =
  match l with | [] -> [] | h::t -> (listReverse t) @ [h];;

let rec digitsOfInt n =
  if n <= 0
  then []
  else ((listReverse n) mod 10) :: (listReverse (digitsOfInt (n / 10)));;


(* fix

let rec listReverse l =
  match l with | [] -> [] | h::t -> (listReverse t) @ [h];;

let rec digitsOfInt n =
  if n <= 0 then [] else (n mod 10) :: (listReverse (digitsOfInt (n / 10)));;

*)

(* changed spans
(8,8)-(8,23)
(8,9)-(8,20)
*)

(* type error slice
(3,2)-(3,57)
(3,36)-(3,51)
(3,37)-(3,48)
(3,49)-(3,50)
(6,5)-(6,6)
(6,5)-(6,11)
(6,10)-(6,11)
(8,8)-(8,23)
(8,9)-(8,20)
(8,21)-(8,22)
*)

(* all spans
(2,20)-(3,57)
(3,2)-(3,57)
(3,8)-(3,9)
(3,23)-(3,25)
(3,36)-(3,57)
(3,52)-(3,53)
(3,36)-(3,51)
(3,37)-(3,48)
(3,49)-(3,50)
(3,54)-(3,57)
(3,55)-(3,56)
(5,20)-(8,71)
(6,2)-(8,71)
(6,5)-(6,11)
(6,5)-(6,6)
(6,10)-(6,11)
(7,7)-(7,9)
(8,7)-(8,71)
(8,7)-(8,31)
(8,8)-(8,23)
(8,9)-(8,20)
(8,21)-(8,22)
(8,28)-(8,30)
(8,35)-(8,71)
(8,36)-(8,47)
(8,48)-(8,70)
(8,49)-(8,60)
(8,61)-(8,69)
(8,62)-(8,63)
(8,66)-(8,68)
*)
