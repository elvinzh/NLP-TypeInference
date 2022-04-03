
let rec appendLists (l1,l2) =
  match l1 with | [] -> l2 | h::t -> h :: (appendLists (t, l2));;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> appendLists ((listReverse t), [h]);;

let palindrome w =
  if (explode w) = (listReverse explode w) then true else false;;


(* fix

let rec appendLists (l1,l2) =
  match l1 with | [] -> l2 | h::t -> h :: (appendLists (t, l2));;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> appendLists ((listReverse t), [h]);;

let palindrome w =
  if (explode w) = (listReverse (explode w)) then true else false;;

*)

(* changed spans
(14,19)-(14,42)
(14,32)-(14,39)
*)

(* type error slice
(5,3)-(8,8)
(5,12)-(8,6)
(11,2)-(11,70)
(11,49)-(11,64)
(11,50)-(11,61)
(11,62)-(11,63)
(14,19)-(14,42)
(14,20)-(14,31)
(14,32)-(14,39)
*)

(* all spans
(2,21)-(3,63)
(3,2)-(3,63)
(3,8)-(3,10)
(3,24)-(3,26)
(3,37)-(3,63)
(3,37)-(3,38)
(3,42)-(3,63)
(3,43)-(3,54)
(3,55)-(3,62)
(3,56)-(3,57)
(3,59)-(3,61)
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
(10,20)-(11,70)
(11,2)-(11,70)
(11,8)-(11,9)
(11,23)-(11,25)
(11,36)-(11,70)
(11,36)-(11,47)
(11,48)-(11,70)
(11,49)-(11,64)
(11,50)-(11,61)
(11,62)-(11,63)
(11,66)-(11,69)
(11,67)-(11,68)
(13,15)-(14,63)
(14,2)-(14,63)
(14,5)-(14,42)
(14,5)-(14,16)
(14,6)-(14,13)
(14,14)-(14,15)
(14,19)-(14,42)
(14,20)-(14,31)
(14,32)-(14,39)
(14,40)-(14,41)
(14,48)-(14,52)
(14,58)-(14,63)
*)
