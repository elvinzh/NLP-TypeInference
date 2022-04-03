
let rec listReverse l =
  match l with | [] -> [] | h::t -> (listReverse t) @ [h];;

let rec digitsOfInt n =
  if n <= 0 then [] else (n mod 10) :: (listReverse digitsOfInt (n / 10));;


(* fix

let rec listReverse l =
  match l with | [] -> [] | h::t -> (listReverse t) @ [h];;

let rec digitsOfInt n =
  if n <= 0 then [] else (n mod 10) :: (listReverse (digitsOfInt (n / 10)));;

*)

(* changed spans
(6,39)-(6,73)
(6,52)-(6,63)
*)

(* type error slice
(3,36)-(3,51)
(3,36)-(3,57)
(3,37)-(3,48)
(3,52)-(3,53)
(6,39)-(6,73)
(6,40)-(6,51)
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
(5,20)-(6,73)
(6,2)-(6,73)
(6,5)-(6,11)
(6,5)-(6,6)
(6,10)-(6,11)
(6,17)-(6,19)
(6,25)-(6,73)
(6,25)-(6,35)
(6,26)-(6,27)
(6,32)-(6,34)
(6,39)-(6,73)
(6,40)-(6,51)
(6,52)-(6,63)
(6,64)-(6,72)
(6,65)-(6,66)
(6,69)-(6,71)
*)