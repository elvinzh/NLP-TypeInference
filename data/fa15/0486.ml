
let rec append xs ys = match xs with | [] -> ys | h::t -> h :: (append t ys);;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> append (listReverse t) [h];;

let palindrome w =
  let l = explode w in
  let lr = listReverse l in if l :: lr then true else false;;


(* fix

let rec append xs ys = match xs with | [] -> ys | h::t -> h :: (append t ys);;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> append (listReverse t) [h];;

let palindrome w =
  let l = explode w in
  let lr = listReverse l in if l = lr then true else false;;

*)

(* changed spans
(14,31)-(14,38)
*)

(* type error slice
(2,23)-(2,76)
(2,45)-(2,47)
(2,58)-(2,59)
(2,58)-(2,76)
(2,63)-(2,76)
(2,64)-(2,70)
(2,71)-(2,72)
(2,73)-(2,75)
(10,2)-(10,62)
(10,36)-(10,42)
(10,36)-(10,62)
(10,43)-(10,58)
(10,44)-(10,55)
(10,56)-(10,57)
(10,59)-(10,62)
(10,60)-(10,61)
(14,2)-(14,59)
(14,11)-(14,22)
(14,11)-(14,24)
(14,23)-(14,24)
(14,28)-(14,59)
(14,31)-(14,32)
(14,31)-(14,38)
(14,36)-(14,38)
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
(4,12)-(7,6)
(5,2)-(7,6)
(5,13)-(6,66)
(6,4)-(6,66)
(6,7)-(6,29)
(6,7)-(6,8)
(6,12)-(6,29)
(6,13)-(6,26)
(6,27)-(6,28)
(6,35)-(6,37)
(6,43)-(6,66)
(6,43)-(6,50)
(6,44)-(6,49)
(6,44)-(6,45)
(6,47)-(6,48)
(6,54)-(6,66)
(6,55)-(6,57)
(6,58)-(6,65)
(6,59)-(6,60)
(6,63)-(6,64)
(7,2)-(7,6)
(7,2)-(7,4)
(7,5)-(7,6)
(9,20)-(10,62)
(10,2)-(10,62)
(10,8)-(10,9)
(10,23)-(10,25)
(10,36)-(10,62)
(10,36)-(10,42)
(10,43)-(10,58)
(10,44)-(10,55)
(10,56)-(10,57)
(10,59)-(10,62)
(10,60)-(10,61)
(12,15)-(14,59)
(13,2)-(14,59)
(13,10)-(13,19)
(13,10)-(13,17)
(13,18)-(13,19)
(14,2)-(14,59)
(14,11)-(14,24)
(14,11)-(14,22)
(14,23)-(14,24)
(14,28)-(14,59)
(14,31)-(14,38)
(14,31)-(14,32)
(14,36)-(14,38)
(14,44)-(14,48)
(14,54)-(14,59)
*)
