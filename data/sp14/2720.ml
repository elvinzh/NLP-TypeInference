
let rec listReverse l =
  match l with | [] -> [] | x::l' -> [listReverse l'; x];;


(* fix

let rec listReverse l =
  match l with | [] -> [] | x::l' -> (listReverse l') @ [x];;

*)

(* changed spans
(3,37)-(3,56)
(3,38)-(3,52)
(3,54)-(3,55)
*)

(* type error slice
(2,3)-(3,58)
(2,20)-(3,56)
(3,2)-(3,56)
(3,37)-(3,56)
(3,38)-(3,49)
(3,38)-(3,52)
*)

(* all spans
(2,20)-(3,56)
(3,2)-(3,56)
(3,8)-(3,9)
(3,23)-(3,25)
(3,37)-(3,56)
(3,38)-(3,52)
(3,38)-(3,49)
(3,50)-(3,52)
(3,54)-(3,55)
*)
