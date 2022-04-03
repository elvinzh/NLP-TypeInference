
let rec listReverse l =
  match l with | [] -> [] | h -> h | h::t -> h @ (listReverse [t]);;


(* fix

let rec listReverse l =
  match l with | [] -> [] | t -> t | h::t -> t @ (listReverse [h]);;

*)

(* changed spans
(3,2)-(3,66)
(3,33)-(3,34)
(3,45)-(3,46)
(3,63)-(3,64)
*)

(* type error slice
(3,2)-(3,66)
(3,33)-(3,34)
(3,45)-(3,46)
(3,45)-(3,66)
(3,47)-(3,48)
*)

(* all spans
(2,20)-(3,66)
(3,2)-(3,66)
(3,8)-(3,9)
(3,23)-(3,25)
(3,33)-(3,34)
(3,45)-(3,66)
(3,47)-(3,48)
(3,45)-(3,46)
(3,49)-(3,66)
(3,50)-(3,61)
(3,62)-(3,65)
(3,63)-(3,64)
*)
