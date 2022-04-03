
let rec digitsOfInt n =
  if n < 0 then [] else (match n with | h::t -> h :: (digitsOfInt t));;


(* fix

let rec digitsOfInt n =
  if n <= 0
  then []
  else (match n with | n_ -> (n_ mod 10) :: (digitsOfInt n_));;

*)

(* changed spans
(3,5)-(3,10)
(3,24)-(3,69)
(3,48)-(3,49)
(3,53)-(3,68)
(3,66)-(3,67)
*)

(* type error slice
(3,5)-(3,6)
(3,5)-(3,10)
(3,9)-(3,10)
(3,24)-(3,69)
(3,31)-(3,32)
*)

(* all spans
(2,20)-(3,69)
(3,2)-(3,69)
(3,5)-(3,10)
(3,5)-(3,6)
(3,9)-(3,10)
(3,16)-(3,18)
(3,24)-(3,69)
(3,31)-(3,32)
(3,48)-(3,68)
(3,48)-(3,49)
(3,53)-(3,68)
(3,54)-(3,65)
(3,66)-(3,67)
*)