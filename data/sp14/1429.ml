
let append x l =
  let rec helper x acc =
    match x with | [] -> acc | h::t -> helper t (h :: acc) in
  helper x l;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> append listReverse t [h];;


(* fix

let append x l =
  let rec helper x acc =
    match x with | [] -> acc | h::t -> helper t (h :: acc) in
  helper x l;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> append (listReverse t) [h];;

*)

(* changed spans
(8,36)-(8,60)
(8,43)-(8,54)
*)

(* type error slice
(2,3)-(5,14)
(2,11)-(5,12)
(2,13)-(5,12)
(3,2)-(5,12)
(4,4)-(4,58)
(4,25)-(4,28)
(4,39)-(4,45)
(4,39)-(4,58)
(4,46)-(4,47)
(4,48)-(4,58)
(4,54)-(4,57)
(5,2)-(5,8)
(5,2)-(5,12)
(5,9)-(5,10)
(7,3)-(8,62)
(7,20)-(8,60)
(8,36)-(8,42)
(8,36)-(8,60)
(8,43)-(8,54)
*)

(* all spans
(2,11)-(5,12)
(2,13)-(5,12)
(3,2)-(5,12)
(3,17)-(4,58)
(3,19)-(4,58)
(4,4)-(4,58)
(4,10)-(4,11)
(4,25)-(4,28)
(4,39)-(4,58)
(4,39)-(4,45)
(4,46)-(4,47)
(4,48)-(4,58)
(4,49)-(4,50)
(4,54)-(4,57)
(5,2)-(5,12)
(5,2)-(5,8)
(5,9)-(5,10)
(5,11)-(5,12)
(7,20)-(8,60)
(8,2)-(8,60)
(8,8)-(8,9)
(8,23)-(8,25)
(8,36)-(8,60)
(8,36)-(8,42)
(8,43)-(8,54)
(8,55)-(8,56)
(8,57)-(8,60)
(8,58)-(8,59)
*)
