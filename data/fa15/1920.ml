
let rec endChar l =
  match l with | [] -> [] | h::[] -> [h] | h::t -> endChar t;;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec removeLast l =
  match l with | [] -> [] | h::[] -> [] | h::t -> h :: (removeLast t);;

let palindrome w =
  let rec palin ls =
    match ls with
    | [] -> true
    | h::[] -> true
    | h::t -> if h = (endChar t) then palin (removeLast t) else false in
  palin (explode w);;


(* fix

let rec endChar l =
  match l with | [] -> [] | h::[] -> [h] | h::t -> endChar t;;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec removeLast l =
  match l with | [] -> [] | h::[] -> [] | h::t -> h :: (removeLast t);;

let palindrome w =
  let rec palin ls =
    match ls with
    | [] -> true
    | h::[] -> true
    | h::t -> if [h] = (endChar t) then palin (removeLast t) else false in
  palin (explode w);;

*)

(* changed spans
(18,17)-(18,18)
*)

(* type error slice
(3,2)-(3,60)
(3,37)-(3,40)
(3,38)-(3,39)
(3,51)-(3,58)
(3,51)-(3,60)
(3,59)-(3,60)
(15,4)-(18,69)
(18,17)-(18,18)
(18,17)-(18,32)
(18,21)-(18,32)
(18,22)-(18,29)
(18,30)-(18,31)
*)

(* all spans
(2,16)-(3,60)
(3,2)-(3,60)
(3,8)-(3,9)
(3,23)-(3,25)
(3,37)-(3,40)
(3,38)-(3,39)
(3,51)-(3,60)
(3,51)-(3,58)
(3,59)-(3,60)
(5,12)-(8,6)
(6,2)-(8,6)
(6,13)-(7,66)
(7,4)-(7,66)
(7,7)-(7,29)
(7,7)-(7,8)
(7,12)-(7,29)
(7,13)-(7,26)
(7,27)-(7,28)
(7,35)-(7,37)
(7,43)-(7,66)
(7,43)-(7,50)
(7,44)-(7,49)
(7,44)-(7,45)
(7,47)-(7,48)
(7,54)-(7,66)
(7,55)-(7,57)
(7,58)-(7,65)
(7,59)-(7,60)
(7,63)-(7,64)
(8,2)-(8,6)
(8,2)-(8,4)
(8,5)-(8,6)
(10,19)-(11,69)
(11,2)-(11,69)
(11,8)-(11,9)
(11,23)-(11,25)
(11,37)-(11,39)
(11,50)-(11,69)
(11,50)-(11,51)
(11,55)-(11,69)
(11,56)-(11,66)
(11,67)-(11,68)
(13,15)-(19,19)
(14,2)-(19,19)
(14,16)-(18,69)
(15,4)-(18,69)
(15,10)-(15,12)
(16,12)-(16,16)
(17,15)-(17,19)
(18,14)-(18,69)
(18,17)-(18,32)
(18,17)-(18,18)
(18,21)-(18,32)
(18,22)-(18,29)
(18,30)-(18,31)
(18,38)-(18,58)
(18,38)-(18,43)
(18,44)-(18,58)
(18,45)-(18,55)
(18,56)-(18,57)
(18,64)-(18,69)
(19,2)-(19,19)
(19,2)-(19,7)
(19,8)-(19,19)
(19,9)-(19,16)
(19,17)-(19,18)
*)
