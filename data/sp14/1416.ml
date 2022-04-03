
let rec digitsOfInt n =
  if n < 0
  then []
  else (match n with | 0 -> [0] | h::t -> [] @ (h @ (digitsOfInt t)));;


(* fix

let rec digitsOfInt n =
  if n < 0
  then []
  else (match n with | 0 -> [0] | _ -> (digitsOfInt (n / 10)) @ [n mod 10]);;

*)

(* changed spans
(5,7)-(5,69)
(5,42)-(5,44)
(5,47)-(5,68)
(5,48)-(5,49)
(5,50)-(5,51)
(5,65)-(5,66)
*)

(* type error slice
(3,5)-(3,6)
(3,5)-(3,10)
(3,9)-(3,10)
(5,7)-(5,69)
(5,14)-(5,15)
*)

(* all spans
(2,20)-(5,69)
(3,2)-(5,69)
(3,5)-(3,10)
(3,5)-(3,6)
(3,9)-(3,10)
(4,7)-(4,9)
(5,7)-(5,69)
(5,14)-(5,15)
(5,28)-(5,31)
(5,29)-(5,30)
(5,42)-(5,68)
(5,45)-(5,46)
(5,42)-(5,44)
(5,47)-(5,68)
(5,50)-(5,51)
(5,48)-(5,49)
(5,52)-(5,67)
(5,53)-(5,64)
(5,65)-(5,66)
*)
