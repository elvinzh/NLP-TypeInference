
let rec listReverse l =
  match l with | [] -> [] | h -> [h] | h::t -> t @ (listReverse h);;


(* fix

let rec listReverse l =
  match l with | [] -> [] | h -> h | h::t -> t @ (listReverse [h]);;

*)

(* changed spans
(3,33)-(3,36)
(3,64)-(3,65)
*)

(* type error slice
(3,2)-(3,66)
(3,33)-(3,36)
(3,34)-(3,35)
(3,47)-(3,48)
(3,47)-(3,66)
(3,49)-(3,50)
*)

(* all spans
(2,20)-(3,66)
(3,2)-(3,66)
(3,8)-(3,9)
(3,23)-(3,25)
(3,33)-(3,36)
(3,34)-(3,35)
(3,47)-(3,66)
(3,49)-(3,50)
(3,47)-(3,48)
(3,51)-(3,66)
(3,52)-(3,63)
(3,64)-(3,65)
*)
