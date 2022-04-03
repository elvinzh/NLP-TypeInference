
let getHead h = match h with | [] -> [] | h::t -> h;;

let getTail t = match t with | [] -> [] | h::t -> t;;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> (listReverse t) @ [h];;

let rec matchHeads x y =
  match x with
  | [] -> true
  | h::t ->
      if (getHead x) = (getHead y) then matchHeads t (getTail y) else false;;

let palindrome w =
  match explode w with
  | [] -> true
  | h::t -> matchHeads (listReverse (explode w)) (listReverse (explode w));;


(* fix

let getHead h = match h with | [] -> [] | h::t -> h;;

let getTail t = match t with | [] -> [] | h::t -> t;;

let rec listReverse l =
  match l with | [] -> [] | h::t -> (listReverse t) @ [h];;

let explode s =
  let rec go i =
    if i >= (String.length s) then [] else (s.[i]) :: (go (i + 1)) in
  go 0;;

let rec matchHeads x =
  match x with
  | [] -> true
  | h::t ->
      if (getHead x) = (getHead (listReverse x))
      then matchHeads (getTail (listReverse t))
      else false;;

let palindrome w =
  match explode w with | [] -> true | h::t -> matchHeads [["b"]];;

*)

(* changed spans
(6,12)-(9,6)
(11,20)-(12,57)
(12,2)-(12,57)
(12,8)-(12,9)
(12,23)-(12,25)
(12,36)-(12,51)
(12,36)-(12,57)
(12,37)-(12,48)
(12,49)-(12,50)
(12,52)-(12,53)
(12,54)-(12,57)
(12,55)-(12,56)
(14,21)-(18,75)
(18,32)-(18,33)
(18,40)-(18,50)
(18,40)-(18,64)
(18,51)-(18,52)
(18,62)-(18,63)
(18,70)-(18,75)
(23,12)-(23,22)
(23,12)-(23,74)
(23,24)-(23,35)
(23,36)-(23,47)
(23,37)-(23,44)
(23,45)-(23,46)
(23,49)-(23,74)
(23,50)-(23,61)
(23,62)-(23,73)
(23,63)-(23,70)
(23,71)-(23,72)
*)

(* type error slice
(2,3)-(2,53)
(2,12)-(2,51)
(2,16)-(2,51)
(2,22)-(2,23)
(2,37)-(2,39)
(2,50)-(2,51)
(6,3)-(9,8)
(6,12)-(9,6)
(7,2)-(9,6)
(8,43)-(8,50)
(8,43)-(8,66)
(8,44)-(8,49)
(8,54)-(8,66)
(8,55)-(8,57)
(9,2)-(9,4)
(9,2)-(9,6)
(12,2)-(12,57)
(12,36)-(12,51)
(12,36)-(12,57)
(12,37)-(12,48)
(12,49)-(12,50)
(12,52)-(12,53)
(12,54)-(12,57)
(12,55)-(12,56)
(15,2)-(18,75)
(15,8)-(15,9)
(18,9)-(18,20)
(18,10)-(18,17)
(18,18)-(18,19)
(18,40)-(18,50)
(18,40)-(18,64)
(18,51)-(18,52)
(23,12)-(23,22)
(23,12)-(23,74)
(23,23)-(23,48)
(23,24)-(23,35)
(23,36)-(23,47)
(23,37)-(23,44)
*)

(* all spans
(2,12)-(2,51)
(2,16)-(2,51)
(2,22)-(2,23)
(2,37)-(2,39)
(2,50)-(2,51)
(4,12)-(4,51)
(4,16)-(4,51)
(4,22)-(4,23)
(4,37)-(4,39)
(4,50)-(4,51)
(6,12)-(9,6)
(7,2)-(9,6)
(7,13)-(8,66)
(8,4)-(8,66)
(8,7)-(8,29)
(8,7)-(8,8)
(8,12)-(8,29)
(8,13)-(8,26)
(8,27)-(8,28)
(8,35)-(8,37)
(8,43)-(8,66)
(8,43)-(8,50)
(8,44)-(8,49)
(8,44)-(8,45)
(8,47)-(8,48)
(8,54)-(8,66)
(8,55)-(8,57)
(8,58)-(8,65)
(8,59)-(8,60)
(8,63)-(8,64)
(9,2)-(9,6)
(9,2)-(9,4)
(9,5)-(9,6)
(11,20)-(12,57)
(12,2)-(12,57)
(12,8)-(12,9)
(12,23)-(12,25)
(12,36)-(12,57)
(12,52)-(12,53)
(12,36)-(12,51)
(12,37)-(12,48)
(12,49)-(12,50)
(12,54)-(12,57)
(12,55)-(12,56)
(14,19)-(18,75)
(14,21)-(18,75)
(15,2)-(18,75)
(15,8)-(15,9)
(16,10)-(16,14)
(18,6)-(18,75)
(18,9)-(18,34)
(18,9)-(18,20)
(18,10)-(18,17)
(18,18)-(18,19)
(18,23)-(18,34)
(18,24)-(18,31)
(18,32)-(18,33)
(18,40)-(18,64)
(18,40)-(18,50)
(18,51)-(18,52)
(18,53)-(18,64)
(18,54)-(18,61)
(18,62)-(18,63)
(18,70)-(18,75)
(20,15)-(23,74)
(21,2)-(23,74)
(21,8)-(21,17)
(21,8)-(21,15)
(21,16)-(21,17)
(22,10)-(22,14)
(23,12)-(23,74)
(23,12)-(23,22)
(23,23)-(23,48)
(23,24)-(23,35)
(23,36)-(23,47)
(23,37)-(23,44)
(23,45)-(23,46)
(23,49)-(23,74)
(23,50)-(23,61)
(23,62)-(23,73)
(23,63)-(23,70)
(23,71)-(23,72)
*)
