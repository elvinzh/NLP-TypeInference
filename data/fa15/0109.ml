
let rec digitsOfInt n =
  let return = [] in
  if n <= 0 then [] :: return else (n mod 10) :: return;
  (digitsOfInt 0) @ return;;


(* fix

let rec digitsOfInt n =
  let return = [] in
  if n <= 0 then return else (n mod 10) :: return; (digitsOfInt 0) @ return;;

*)

(* changed spans
(4,17)-(4,19)
(4,17)-(4,29)
*)

(* type error slice
(4,17)-(4,19)
(4,17)-(4,29)
(4,23)-(4,29)
(4,35)-(4,45)
(4,35)-(4,55)
(4,49)-(4,55)
*)

(* all spans
(2,20)-(5,26)
(3,2)-(5,26)
(3,15)-(3,17)
(4,2)-(5,26)
(4,2)-(4,55)
(4,5)-(4,11)
(4,5)-(4,6)
(4,10)-(4,11)
(4,17)-(4,29)
(4,17)-(4,19)
(4,23)-(4,29)
(4,35)-(4,55)
(4,35)-(4,45)
(4,36)-(4,37)
(4,42)-(4,44)
(4,49)-(4,55)
(5,2)-(5,26)
(5,18)-(5,19)
(5,2)-(5,17)
(5,3)-(5,14)
(5,15)-(5,16)
(5,20)-(5,26)
*)
