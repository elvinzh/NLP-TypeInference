
let rec helperAppend l n =
  match l with | [] -> n | h::t -> h :: (helperAppend t n);;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> helperAppend (listReverse t) [h];;

let palindrome w = (listReverse (explode w)) = w;;


(* fix

let rec helperAppend l n =
  match l with | [] -> n | h::t -> h :: (helperAppend t n);;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> helperAppend (listReverse t) [h];;

let palindrome w = (listReverse (explode w)) = (explode w);;

*)

(* changed spans
(13,47)-(13,48)
*)

(* type error slice
(3,2)-(3,58)
(3,40)-(3,58)
(3,41)-(3,53)
(3,54)-(3,55)
(5,3)-(8,8)
(5,12)-(8,6)
(7,12)-(7,29)
(7,13)-(7,26)
(7,27)-(7,28)
(11,36)-(11,48)
(11,36)-(11,68)
(11,49)-(11,64)
(11,50)-(11,61)
(13,19)-(13,44)
(13,19)-(13,48)
(13,20)-(13,31)
(13,32)-(13,43)
(13,33)-(13,40)
(13,41)-(13,42)
(13,47)-(13,48)
*)

(* all spans
(2,21)-(3,58)
(2,23)-(3,58)
(3,2)-(3,58)
(3,8)-(3,9)
(3,23)-(3,24)
(3,35)-(3,58)
(3,35)-(3,36)
(3,40)-(3,58)
(3,41)-(3,53)
(3,54)-(3,55)
(3,56)-(3,57)
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
(10,20)-(11,68)
(11,2)-(11,68)
(11,8)-(11,9)
(11,23)-(11,25)
(11,36)-(11,68)
(11,36)-(11,48)
(11,49)-(11,64)
(11,50)-(11,61)
(11,62)-(11,63)
(11,65)-(11,68)
(11,66)-(11,67)
(13,15)-(13,48)
(13,19)-(13,48)
(13,19)-(13,44)
(13,20)-(13,31)
(13,32)-(13,43)
(13,33)-(13,40)
(13,41)-(13,42)
(13,47)-(13,48)
*)
