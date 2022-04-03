
let rec append l1 l2 = match l1 with | [] -> l2 | h::t -> h :: (append t l2);;

let rec listReverse l =
  match l with | [] -> [] | h::t -> append (listReverse t [h]);;


(* fix

let rec append l1 l2 = match l1 with | [] -> l2 | h::t -> h :: (append t l2);;

let rec listReverse l = match l with | [] -> [] | h::t -> append [h] [h];;

*)

(* changed spans
(5,36)-(5,42)
(5,36)-(5,62)
(5,44)-(5,55)
(5,56)-(5,57)
*)

(* type error slice
(2,63)-(2,76)
(2,64)-(2,70)
(4,3)-(5,64)
(4,20)-(5,62)
(5,2)-(5,62)
(5,23)-(5,25)
(5,36)-(5,42)
(5,36)-(5,62)
(5,43)-(5,62)
(5,44)-(5,55)
*)

(* all spans
(2,15)-(2,76)
(2,18)-(2,76)
(2,23)-(2,76)
(2,29)-(2,31)
(2,45)-(2,47)
(2,58)-(2,76)
(2,58)-(2,59)
(2,63)-(2,76)
(2,64)-(2,70)
(2,71)-(2,72)
(2,73)-(2,75)
(4,20)-(5,62)
(5,2)-(5,62)
(5,8)-(5,9)
(5,23)-(5,25)
(5,36)-(5,62)
(5,36)-(5,42)
(5,43)-(5,62)
(5,44)-(5,55)
(5,56)-(5,57)
(5,58)-(5,61)
(5,59)-(5,60)
*)
