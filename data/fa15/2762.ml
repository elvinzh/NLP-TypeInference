
let remainder x y = if (x * y) > 10 then (x * y) mod 10 else 0;;

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      if (List.length t) = 0
      then [h * i] @ (mulByDigit i t)
      else [remainder i h] :: (mulByDigit i t);;


(* fix

let remainder x y = if (x * y) > 10 then (x * y) mod 10 else 0;;

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      if (List.length t) = 0
      then [h * i] @ (mulByDigit i t)
      else (remainder i h) :: (mulByDigit i t);;

*)

(* changed spans
(10,11)-(10,26)
*)

(* type error slice
(9,11)-(9,18)
(9,11)-(9,37)
(9,12)-(9,17)
(9,19)-(9,20)
(9,21)-(9,37)
(9,22)-(9,32)
(10,11)-(10,26)
(10,11)-(10,46)
(10,30)-(10,46)
(10,31)-(10,41)
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
(4,19)-(10,46)
(4,21)-(10,46)
(5,2)-(10,46)
(5,8)-(5,18)
(5,8)-(5,16)
(5,17)-(5,18)
(6,10)-(6,12)
(8,6)-(10,46)
(8,9)-(8,28)
(8,9)-(8,24)
(8,10)-(8,21)
(8,22)-(8,23)
(8,27)-(8,28)
(9,11)-(9,37)
(9,19)-(9,20)
(9,11)-(9,18)
(9,12)-(9,17)
(9,12)-(9,13)
(9,16)-(9,17)
(9,21)-(9,37)
(9,22)-(9,32)
(9,33)-(9,34)
(9,35)-(9,36)
(10,11)-(10,46)
(10,11)-(10,26)
(10,12)-(10,25)
(10,12)-(10,21)
(10,22)-(10,23)
(10,24)-(10,25)
(10,30)-(10,46)
(10,31)-(10,41)
(10,42)-(10,43)
(10,44)-(10,45)
*)
