
let remainder x y = if (x * y) > 10 then (x * y) mod 10 else 0;;

let x l = List.map string_of_int;;

let y = [1; 2; 3];;

let rec mulByDigit i l =
  if i <= 0
  then []
  else
    (match l with
     | [] -> []
     | h::t ->
         if (List.length t) > 1
         then (remainder i h) :: (mulByDigit i t)
         else [x * y]);;


(* fix

let remainder x y = if (x * y) > 10 then (x * y) mod 10 else 0;;

let rec mulByDigit i l =
  if i <= 0
  then []
  else
    (match l with
     | [] -> []
     | h::t ->
         if (List.length t) > 1
         then (remainder i h) :: (mulByDigit i t)
         else [i * h]);;

*)

(* changed spans
(4,6)-(4,32)
(4,10)-(4,18)
(4,10)-(4,32)
(4,19)-(4,32)
(6,8)-(6,17)
(6,9)-(6,10)
(6,12)-(6,13)
(6,15)-(6,16)
(17,15)-(17,16)
(17,19)-(17,20)
*)

(* type error slice
(4,3)-(4,34)
(4,6)-(4,32)
(6,3)-(6,19)
(6,8)-(6,17)
(17,15)-(17,16)
(17,15)-(17,20)
(17,19)-(17,20)
*)

(* all spans
(2,14)-(2,62)
(2,16)-(2,62)
(2,20)-(2,62)
(2,23)-(2,35)
(2,23)-(2,30)
(2,24)-(2,25)
(2,28)-(2,29)
(2,33)-(2,35)
(2,41)-(2,55)
(2,41)-(2,48)
(2,42)-(2,43)
(2,46)-(2,47)
(2,53)-(2,55)
(2,61)-(2,62)
(4,6)-(4,32)
(4,10)-(4,32)
(4,10)-(4,18)
(4,19)-(4,32)
(6,8)-(6,17)
(6,9)-(6,10)
(6,12)-(6,13)
(6,15)-(6,16)
(8,19)-(17,22)
(8,21)-(17,22)
(9,2)-(17,22)
(9,5)-(9,11)
(9,5)-(9,6)
(9,10)-(9,11)
(10,7)-(10,9)
(12,4)-(17,22)
(12,11)-(12,12)
(13,13)-(13,15)
(15,9)-(17,21)
(15,12)-(15,31)
(15,12)-(15,27)
(15,13)-(15,24)
(15,25)-(15,26)
(15,30)-(15,31)
(16,14)-(16,49)
(16,14)-(16,29)
(16,15)-(16,24)
(16,25)-(16,26)
(16,27)-(16,28)
(16,33)-(16,49)
(16,34)-(16,44)
(16,45)-(16,46)
(16,47)-(16,48)
(17,14)-(17,21)
(17,15)-(17,20)
(17,15)-(17,16)
(17,19)-(17,20)
*)
