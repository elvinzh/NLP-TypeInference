
let rec digitsOfInt n =
  let return = [] in
  if n < 0 then return else (digitsOfInt (-1) (n mod 10)) :: return;;


(* fix

let rec digitsOfInt n =
  let return = [] in if n < 0 then return else (n mod 10) :: return;;

*)

(* changed spans
(4,28)-(4,57)
(4,29)-(4,40)
(4,41)-(4,45)
*)

(* type error slice
(2,3)-(4,69)
(2,20)-(4,67)
(3,2)-(4,67)
(4,2)-(4,67)
(4,28)-(4,57)
(4,28)-(4,67)
(4,29)-(4,40)
*)

(* all spans
(2,20)-(4,67)
(3,2)-(4,67)
(3,15)-(3,17)
(4,2)-(4,67)
(4,5)-(4,10)
(4,5)-(4,6)
(4,9)-(4,10)
(4,16)-(4,22)
(4,28)-(4,67)
(4,28)-(4,57)
(4,29)-(4,40)
(4,41)-(4,45)
(4,46)-(4,56)
(4,47)-(4,48)
(4,53)-(4,55)
(4,61)-(4,67)
*)
