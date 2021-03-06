
let rec clone x n =
  if n < 1 then [] else (match n with | _ -> x :: (clone x (n - 1)));;

let c y = y;;

let padZero l1 l2 =
  let s1 = List.length l1 in
  let s2 = List.length l2 in
  if s1 = s2
  then (l1, l2)
  else
    if s1 > s2
    then (l1, ((clone 0 (s1 - s2)) @ l2))
    else (((clone 0 (s2 - s1)) @ l1), l2);;

let rec removeZero l =
  if l = []
  then []
  else (let h::t = l in match h with | 0 -> removeZero t | _ -> l);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x1,x2) = x in
      let (a1,a2) = a in
      let s::s' = a2 in
      ((((x1 + x2) + a1) / 10), ([s + c] @ (s' @ [((x1 + x2) + c) mod 10]))) in
    let base = (0, [0]) in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n =
  if n < 1 then [] else (match n with | _ -> x :: (clone x (n - 1)));;

let padZero l1 l2 =
  let s1 = List.length l1 in
  let s2 = List.length l2 in
  if s1 = s2
  then (l1, l2)
  else
    if s1 > s2
    then (l1, ((clone 0 (s1 - s2)) @ l2))
    else (((clone 0 (s2 - s1)) @ l1), l2);;

let rec removeZero l =
  if l = []
  then []
  else (let h::t = l in match h with | 0 -> removeZero t | _ -> l);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x1,x2) = x in
      let (c,a2) = a in
      let s::s' = a2 in
      ((((x1 + x2) + c) / 10), ([s + c] @ (s' @ [((x1 + x2) + c) mod 10]))) in
    let base = (0, [0]) in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(5,6)-(5,11)
(5,10)-(5,11)
(26,6)-(28,76)
(28,21)-(28,23)
*)

(* type error slice
(5,3)-(5,13)
(5,6)-(5,11)
(28,34)-(28,39)
(28,38)-(28,39)
(28,50)-(28,65)
(28,63)-(28,64)
*)

(* all spans
(2,14)-(3,68)
(2,16)-(3,68)
(3,2)-(3,68)
(3,5)-(3,10)
(3,5)-(3,6)
(3,9)-(3,10)
(3,16)-(3,18)
(3,24)-(3,68)
(3,31)-(3,32)
(3,45)-(3,67)
(3,45)-(3,46)
(3,50)-(3,67)
(3,51)-(3,56)
(3,57)-(3,58)
(3,59)-(3,66)
(3,60)-(3,61)
(3,64)-(3,65)
(5,6)-(5,11)
(5,10)-(5,11)
(7,12)-(15,41)
(7,15)-(15,41)
(8,2)-(15,41)
(8,11)-(8,25)
(8,11)-(8,22)
(8,23)-(8,25)
(9,2)-(15,41)
(9,11)-(9,25)
(9,11)-(9,22)
(9,23)-(9,25)
(10,2)-(15,41)
(10,5)-(10,12)
(10,5)-(10,7)
(10,10)-(10,12)
(11,7)-(11,15)
(11,8)-(11,10)
(11,12)-(11,14)
(13,4)-(15,41)
(13,7)-(13,14)
(13,7)-(13,9)
(13,12)-(13,14)
(14,9)-(14,41)
(14,10)-(14,12)
(14,14)-(14,40)
(14,35)-(14,36)
(14,15)-(14,34)
(14,16)-(14,21)
(14,22)-(14,23)
(14,24)-(14,33)
(14,25)-(14,27)
(14,30)-(14,32)
(14,37)-(14,39)
(15,9)-(15,41)
(15,10)-(15,36)
(15,31)-(15,32)
(15,11)-(15,30)
(15,12)-(15,17)
(15,18)-(15,19)
(15,20)-(15,29)
(15,21)-(15,23)
(15,26)-(15,28)
(15,33)-(15,35)
(15,38)-(15,40)
(17,19)-(20,66)
(18,2)-(20,66)
(18,5)-(18,11)
(18,5)-(18,6)
(18,9)-(18,11)
(19,7)-(19,9)
(20,7)-(20,66)
(20,19)-(20,20)
(20,24)-(20,65)
(20,30)-(20,31)
(20,44)-(20,56)
(20,44)-(20,54)
(20,55)-(20,56)
(20,64)-(20,65)
(22,11)-(32,34)
(22,14)-(32,34)
(23,2)-(32,34)
(23,11)-(31,51)
(24,4)-(31,51)
(24,10)-(28,76)
(24,12)-(28,76)
(25,6)-(28,76)
(25,20)-(25,21)
(26,6)-(28,76)
(26,20)-(26,21)
(27,6)-(28,76)
(27,18)-(27,20)
(28,6)-(28,76)
(28,7)-(28,30)
(28,8)-(28,24)
(28,9)-(28,18)
(28,10)-(28,12)
(28,15)-(28,17)
(28,21)-(28,23)
(28,27)-(28,29)
(28,32)-(28,75)
(28,41)-(28,42)
(28,33)-(28,40)
(28,34)-(28,39)
(28,34)-(28,35)
(28,38)-(28,39)
(28,43)-(28,74)
(28,47)-(28,48)
(28,44)-(28,46)
(28,49)-(28,73)
(28,50)-(28,72)
(28,50)-(28,65)
(28,51)-(28,60)
(28,52)-(28,54)
(28,57)-(28,59)
(28,63)-(28,64)
(28,70)-(28,72)
(29,4)-(31,51)
(29,15)-(29,23)
(29,16)-(29,17)
(29,19)-(29,22)
(29,20)-(29,21)
(30,4)-(31,51)
(30,15)-(30,33)
(30,15)-(30,27)
(30,28)-(30,30)
(30,31)-(30,33)
(31,4)-(31,51)
(31,18)-(31,44)
(31,18)-(31,32)
(31,33)-(31,34)
(31,35)-(31,39)
(31,40)-(31,44)
(31,48)-(31,51)
(32,2)-(32,34)
(32,2)-(32,12)
(32,13)-(32,34)
(32,14)-(32,17)
(32,18)-(32,33)
(32,19)-(32,26)
(32,27)-(32,29)
(32,30)-(32,32)
*)
