
let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      let prod = h * i in
      if prod > 10
      then [prod mod 10; (prod / 10) :: (mulByDigit i t)]
      else prod :: t;;


(* fix

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      let prod = h * i in
      if prod > 10
      then (prod mod 10) :: (prod / 10) :: (mulByDigit i t)
      else (prod mod 10) :: t;;

*)

(* changed spans
(8,11)-(8,57)
(9,11)-(9,15)
(9,19)-(9,20)
*)

(* type error slice
(8,11)-(8,57)
(8,12)-(8,23)
(8,25)-(8,56)
*)

(* all spans
(2,19)-(9,20)
(2,21)-(9,20)
(3,2)-(9,20)
(3,8)-(3,18)
(3,8)-(3,16)
(3,17)-(3,18)
(4,10)-(4,12)
(6,6)-(9,20)
(6,17)-(6,22)
(6,17)-(6,18)
(6,21)-(6,22)
(7,6)-(9,20)
(7,9)-(7,18)
(7,9)-(7,13)
(7,16)-(7,18)
(8,11)-(8,57)
(8,12)-(8,23)
(8,12)-(8,16)
(8,21)-(8,23)
(8,25)-(8,56)
(8,25)-(8,36)
(8,26)-(8,30)
(8,33)-(8,35)
(8,40)-(8,56)
(8,41)-(8,51)
(8,52)-(8,53)
(8,54)-(8,55)
(9,11)-(9,20)
(9,11)-(9,15)
(9,19)-(9,20)
*)
