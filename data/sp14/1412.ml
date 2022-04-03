
let rec digitsOfInt n =
  if n < 0
  then []
  else (match n / 10 with | 0 -> [0] | _ -> digitsOfInt n []);;


(* fix

let rec digitsOfInt n =
  if n < 0 then [] else (match n / 10 with | 0 -> [0] | _ -> digitsOfInt n);;

*)

(* changed spans
(5,44)-(5,60)
(5,58)-(5,60)
*)

(* type error slice
(2,3)-(5,63)
(2,20)-(5,61)
(3,2)-(5,61)
(5,7)-(5,61)
(5,44)-(5,55)
(5,44)-(5,60)
*)

(* all spans
(2,20)-(5,61)
(3,2)-(5,61)
(3,5)-(3,10)
(3,5)-(3,6)
(3,9)-(3,10)
(4,7)-(4,9)
(5,7)-(5,61)
(5,14)-(5,20)
(5,14)-(5,15)
(5,18)-(5,20)
(5,33)-(5,36)
(5,34)-(5,35)
(5,44)-(5,60)
(5,44)-(5,55)
(5,56)-(5,57)
(5,58)-(5,60)
*)
