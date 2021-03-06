
let rec mulByDigit i l =
  let rec mBDhelper i x =
    match x with
    | [] -> []
    | hd::tl ->
        if ((hd * i) / 10) != 0
        then ((hd * i) mod 10) :: (((hd * i) / 10) + (mBDhelper i tl))
        else (hd * i) :: (mBDhelper i tl) in
  mBDhelper i l;;


(* fix

let rec mulByDigit i l =
  let comb a b = match b with | [] -> [a] | hd::tl -> [a + hd] in
  let rec mBDhelper i x =
    match x with
    | [] -> []
    | hd::tl ->
        if ((hd * i) - 9) != 0
        then ((hd * i) / 10) :: (comb ((hd * i) mod 10) (mBDhelper i tl))
        else (hd * i) :: (mBDhelper i tl) in
  mBDhelper i l;;

*)

(* changed spans
(3,2)-(10,15)
(7,11)-(7,26)
(7,23)-(7,25)
(8,13)-(8,30)
(8,34)-(8,70)
(8,35)-(8,50)
(8,36)-(8,44)
*)

(* type error slice
(3,2)-(10,15)
(3,20)-(9,41)
(3,22)-(9,41)
(4,4)-(9,41)
(7,8)-(9,41)
(8,13)-(8,70)
(8,34)-(8,70)
(8,53)-(8,69)
(8,54)-(8,63)
(9,13)-(9,41)
(9,25)-(9,41)
(9,26)-(9,35)
*)

(* all spans
(2,19)-(10,15)
(2,21)-(10,15)
(3,2)-(10,15)
(3,20)-(9,41)
(3,22)-(9,41)
(4,4)-(9,41)
(4,10)-(4,11)
(5,12)-(5,14)
(7,8)-(9,41)
(7,11)-(7,31)
(7,11)-(7,26)
(7,12)-(7,20)
(7,13)-(7,15)
(7,18)-(7,19)
(7,23)-(7,25)
(7,30)-(7,31)
(8,13)-(8,70)
(8,13)-(8,30)
(8,14)-(8,22)
(8,15)-(8,17)
(8,20)-(8,21)
(8,27)-(8,29)
(8,34)-(8,70)
(8,35)-(8,50)
(8,36)-(8,44)
(8,37)-(8,39)
(8,42)-(8,43)
(8,47)-(8,49)
(8,53)-(8,69)
(8,54)-(8,63)
(8,64)-(8,65)
(8,66)-(8,68)
(9,13)-(9,41)
(9,13)-(9,21)
(9,14)-(9,16)
(9,19)-(9,20)
(9,25)-(9,41)
(9,26)-(9,35)
(9,36)-(9,37)
(9,38)-(9,40)
(10,2)-(10,15)
(10,2)-(10,11)
(10,12)-(10,13)
(10,14)-(10,15)
*)
