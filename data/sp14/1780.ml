
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let l1 = [9; 9; 9];;

let rec mulByDigit i l =
  let f a x =
    let (i,j) = x in
    let (s,t) = a in ((((i * j) + s) / 10), ((((i * j) + s) mod 10) :: t)) in
  let base = (0, []) in
  let args =
    List.combine (List.rev (0 :: l1)) (clone i ((List.length + 1) l)) in
  let (_,res) = List.fold_left f base args in res;;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let l1 = [9; 9; 9];;

let rec mulByDigit i l =
  let f a x =
    let (i,j) = x in
    let (s,t) = a in ((((i * j) + s) / 10), ((((i * j) + s) mod 10) :: t)) in
  let base = (0, []) in
  let args =
    List.combine (List.rev (0 :: l1)) (clone i ((List.length l) + 1)) in
  let (_,res) = List.fold_left f base args in res;;

*)

(* changed spans
(12,47)-(12,68)
(12,49)-(12,60)
(12,63)-(12,64)
(13,2)-(13,49)
*)

(* type error slice
(12,47)-(12,68)
(12,48)-(12,65)
(12,49)-(12,60)
*)

(* all spans
(2,14)-(2,65)
(2,16)-(2,65)
(2,20)-(2,65)
(2,23)-(2,29)
(2,23)-(2,24)
(2,28)-(2,29)
(2,35)-(2,37)
(2,43)-(2,65)
(2,43)-(2,44)
(2,48)-(2,65)
(2,49)-(2,54)
(2,55)-(2,56)
(2,57)-(2,64)
(2,58)-(2,59)
(2,62)-(2,63)
(4,9)-(4,18)
(4,10)-(4,11)
(4,13)-(4,14)
(4,16)-(4,17)
(6,19)-(13,49)
(6,21)-(13,49)
(7,2)-(13,49)
(7,8)-(9,74)
(7,10)-(9,74)
(8,4)-(9,74)
(8,16)-(8,17)
(9,4)-(9,74)
(9,16)-(9,17)
(9,21)-(9,74)
(9,22)-(9,42)
(9,23)-(9,36)
(9,24)-(9,31)
(9,25)-(9,26)
(9,29)-(9,30)
(9,34)-(9,35)
(9,39)-(9,41)
(9,44)-(9,73)
(9,45)-(9,67)
(9,46)-(9,59)
(9,47)-(9,54)
(9,48)-(9,49)
(9,52)-(9,53)
(9,57)-(9,58)
(9,64)-(9,66)
(9,71)-(9,72)
(10,2)-(13,49)
(10,13)-(10,20)
(10,14)-(10,15)
(10,17)-(10,19)
(11,2)-(13,49)
(12,4)-(12,69)
(12,4)-(12,16)
(12,17)-(12,37)
(12,18)-(12,26)
(12,27)-(12,36)
(12,28)-(12,29)
(12,33)-(12,35)
(12,38)-(12,69)
(12,39)-(12,44)
(12,45)-(12,46)
(12,47)-(12,68)
(12,48)-(12,65)
(12,49)-(12,60)
(12,63)-(12,64)
(12,66)-(12,67)
(13,2)-(13,49)
(13,16)-(13,42)
(13,16)-(13,30)
(13,31)-(13,32)
(13,33)-(13,37)
(13,38)-(13,42)
(13,46)-(13,49)
*)
