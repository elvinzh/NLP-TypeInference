
let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> (listReverse t) @ [h];;

let palindrome w = (listReverse (explode w)) == w;;


(* fix

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> (listReverse t) @ [h];;

let palindrome w = (listReverse (explode w)) == (explode w);;

*)

(* changed spans
(10,48)-(10,49)
*)

(* type error slice
(2,3)-(5,8)
(2,12)-(5,6)
(4,12)-(4,29)
(4,13)-(4,26)
(4,27)-(4,28)
(8,36)-(8,51)
(8,36)-(8,57)
(8,37)-(8,48)
(8,52)-(8,53)
(10,19)-(10,44)
(10,19)-(10,49)
(10,20)-(10,31)
(10,32)-(10,43)
(10,33)-(10,40)
(10,41)-(10,42)
(10,48)-(10,49)
*)

(* all spans
(2,12)-(5,6)
(3,2)-(5,6)
(3,13)-(4,66)
(4,4)-(4,66)
(4,7)-(4,29)
(4,7)-(4,8)
(4,12)-(4,29)
(4,13)-(4,26)
(4,27)-(4,28)
(4,35)-(4,37)
(4,43)-(4,66)
(4,43)-(4,50)
(4,44)-(4,49)
(4,44)-(4,45)
(4,47)-(4,48)
(4,54)-(4,66)
(4,55)-(4,57)
(4,58)-(4,65)
(4,59)-(4,60)
(4,63)-(4,64)
(5,2)-(5,6)
(5,2)-(5,4)
(5,5)-(5,6)
(7,20)-(8,57)
(8,2)-(8,57)
(8,8)-(8,9)
(8,23)-(8,25)
(8,36)-(8,57)
(8,52)-(8,53)
(8,36)-(8,51)
(8,37)-(8,48)
(8,49)-(8,50)
(8,54)-(8,57)
(8,55)-(8,56)
(10,15)-(10,49)
(10,19)-(10,49)
(10,19)-(10,44)
(10,20)-(10,31)
(10,32)-(10,43)
(10,33)-(10,40)
(10,41)-(10,42)
(10,48)-(10,49)
*)