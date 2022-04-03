
let rec digitsOfInt n =
  if n < 0
  then []
  else if n < 10 then [n] else [digitsOfInt (n / 10); n mod 10];;


(* fix

let rec digitsOfInt n =
  if n < 0
  then []
  else
    if n = 0
    then [0]
    else
      if n > 10
      then digitsOfInt (n mod 10)
      else (let a = n mod 10 in let b = n / 10 in if b = 0 then [n] else [a]);;

*)

(* changed spans
(5,10)-(5,16)
(5,14)-(5,16)
(5,23)-(5,24)
(5,31)-(5,63)
(5,44)-(5,52)
(5,54)-(5,62)
*)

(* type error slice
(2,3)-(5,65)
(2,20)-(5,63)
(3,2)-(5,63)
(5,7)-(5,63)
(5,31)-(5,63)
(5,32)-(5,43)
(5,32)-(5,52)
*)

(* all spans
(2,20)-(5,63)
(3,2)-(5,63)
(3,5)-(3,10)
(3,5)-(3,6)
(3,9)-(3,10)
(4,7)-(4,9)
(5,7)-(5,63)
(5,10)-(5,16)
(5,10)-(5,11)
(5,14)-(5,16)
(5,22)-(5,25)
(5,23)-(5,24)
(5,31)-(5,63)
(5,32)-(5,52)
(5,32)-(5,43)
(5,44)-(5,52)
(5,45)-(5,46)
(5,49)-(5,51)
(5,54)-(5,62)
(5,54)-(5,55)
(5,60)-(5,62)
*)
