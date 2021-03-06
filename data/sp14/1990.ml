
let rec myAppend l n = match l with | [] -> [n] | h::t -> h :: (myAppend t n);;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listCompare l k =
  if ((List.hd l) = []) && ((List.hd k) = [])
  then true
  else
    if (List.hd l) = (List.hd k)
    then listCompare (List.tl l) (List.tl k)
    else false;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> myAppend (listReverse t) h;;

let palindrome w = listCompare (explode w) (listReverse (explode w));;


(* fix

let rec myAppend l n = match l with | [] -> [n] | h::t -> h :: (myAppend t n);;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> myAppend (listReverse t) h;;

let palindrome w = (explode w) = (listReverse (explode w));;

*)

(* changed spans
(9,22)-(15,14)
(10,2)-(15,14)
(10,5)-(10,23)
(10,5)-(10,45)
(10,6)-(10,17)
(10,7)-(10,14)
(10,15)-(10,16)
(10,20)-(10,22)
(10,27)-(10,45)
(10,28)-(10,39)
(10,29)-(10,36)
(10,37)-(10,38)
(10,42)-(10,44)
(11,7)-(11,11)
(13,4)-(15,14)
(13,7)-(13,18)
(13,7)-(13,32)
(13,8)-(13,15)
(13,16)-(13,17)
(13,21)-(13,32)
(13,22)-(13,29)
(13,30)-(13,31)
(14,9)-(14,20)
(14,9)-(14,44)
(14,21)-(14,32)
(14,22)-(14,29)
(14,30)-(14,31)
(14,33)-(14,44)
(14,34)-(14,41)
(14,42)-(14,43)
(15,9)-(15,14)
(17,20)-(18,62)
(20,19)-(20,30)
(20,19)-(20,68)
*)

(* type error slice
(4,3)-(7,8)
(4,12)-(7,6)
(5,2)-(7,6)
(6,43)-(6,50)
(6,43)-(6,66)
(6,44)-(6,49)
(6,54)-(6,66)
(6,55)-(6,57)
(7,2)-(7,4)
(7,2)-(7,6)
(10,27)-(10,45)
(10,28)-(10,39)
(10,29)-(10,36)
(10,37)-(10,38)
(10,42)-(10,44)
(13,7)-(13,18)
(13,7)-(13,32)
(13,8)-(13,15)
(13,16)-(13,17)
(13,21)-(13,32)
(13,22)-(13,29)
(13,30)-(13,31)
(14,9)-(14,20)
(14,9)-(14,44)
(14,21)-(14,32)
(14,22)-(14,29)
(14,30)-(14,31)
(20,19)-(20,30)
(20,19)-(20,68)
(20,31)-(20,42)
(20,32)-(20,39)
*)

(* all spans
(2,17)-(2,77)
(2,19)-(2,77)
(2,23)-(2,77)
(2,29)-(2,30)
(2,44)-(2,47)
(2,45)-(2,46)
(2,58)-(2,77)
(2,58)-(2,59)
(2,63)-(2,77)
(2,64)-(2,72)
(2,73)-(2,74)
(2,75)-(2,76)
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
(9,20)-(15,14)
(9,22)-(15,14)
(10,2)-(15,14)
(10,5)-(10,45)
(10,5)-(10,23)
(10,6)-(10,17)
(10,7)-(10,14)
(10,15)-(10,16)
(10,20)-(10,22)
(10,27)-(10,45)
(10,28)-(10,39)
(10,29)-(10,36)
(10,37)-(10,38)
(10,42)-(10,44)
(11,7)-(11,11)
(13,4)-(15,14)
(13,7)-(13,32)
(13,7)-(13,18)
(13,8)-(13,15)
(13,16)-(13,17)
(13,21)-(13,32)
(13,22)-(13,29)
(13,30)-(13,31)
(14,9)-(14,44)
(14,9)-(14,20)
(14,21)-(14,32)
(14,22)-(14,29)
(14,30)-(14,31)
(14,33)-(14,44)
(14,34)-(14,41)
(14,42)-(14,43)
(15,9)-(15,14)
(17,20)-(18,62)
(18,2)-(18,62)
(18,8)-(18,9)
(18,23)-(18,25)
(18,36)-(18,62)
(18,36)-(18,44)
(18,45)-(18,60)
(18,46)-(18,57)
(18,58)-(18,59)
(18,61)-(18,62)
(20,15)-(20,68)
(20,19)-(20,68)
(20,19)-(20,30)
(20,31)-(20,42)
(20,32)-(20,39)
(20,40)-(20,41)
(20,43)-(20,68)
(20,44)-(20,55)
(20,56)-(20,67)
(20,57)-(20,64)
(20,65)-(20,66)
*)
