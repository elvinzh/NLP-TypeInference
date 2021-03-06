
let rec clone x n =
  if n <= 0 then [] else if n = 1 then [x] else [x] @ (clone x (n - 1));;

let padZero l1 l2 =
  let n = (List.length l1) - (List.length l2) in
  if n < 0 then (((clone 0 (- n)) @ l1), l2) else (l1, ((clone 0 n) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x1,x2) = x in
      let (a1,a2) = a in
      if ((a1 + x1) + x2) >= 10 then 1 else (0, (((a1 + x1) + x2) :: a2)) in
    let base = [(0, 0)] in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n =
  if n <= 0 then [] else if n = 1 then [x] else [x] @ (clone x (n - 1));;

let padZero l1 l2 =
  let n = (List.length l1) - (List.length l2) in
  if n < 0 then (((clone 0 (- n)) @ l1), l2) else (l1, ((clone 0 n) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x1,x2) = x in
      let (a1,a2) = a in
      ((if ((a1 + x1) + x2) >= 10 then 1 else 0), (((a1 + x1) + x2) :: a2)) in
    let base = (0, [0]) in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(17,6)-(17,73)
(17,44)-(17,73)
(18,15)-(18,23)
(18,20)-(18,21)
*)

(* type error slice
(17,6)-(17,73)
(17,37)-(17,38)
(17,44)-(17,73)
*)

(* all spans
(2,14)-(3,71)
(2,16)-(3,71)
(3,2)-(3,71)
(3,5)-(3,11)
(3,5)-(3,6)
(3,10)-(3,11)
(3,17)-(3,19)
(3,25)-(3,71)
(3,28)-(3,33)
(3,28)-(3,29)
(3,32)-(3,33)
(3,39)-(3,42)
(3,40)-(3,41)
(3,48)-(3,71)
(3,52)-(3,53)
(3,48)-(3,51)
(3,49)-(3,50)
(3,54)-(3,71)
(3,55)-(3,60)
(3,61)-(3,62)
(3,63)-(3,70)
(3,64)-(3,65)
(3,68)-(3,69)
(5,12)-(7,74)
(5,15)-(7,74)
(6,2)-(7,74)
(6,10)-(6,45)
(6,10)-(6,26)
(6,11)-(6,22)
(6,23)-(6,25)
(6,29)-(6,45)
(6,30)-(6,41)
(6,42)-(6,44)
(7,2)-(7,74)
(7,5)-(7,10)
(7,5)-(7,6)
(7,9)-(7,10)
(7,16)-(7,44)
(7,17)-(7,39)
(7,34)-(7,35)
(7,18)-(7,33)
(7,19)-(7,24)
(7,25)-(7,26)
(7,27)-(7,32)
(7,30)-(7,31)
(7,36)-(7,38)
(7,41)-(7,43)
(7,50)-(7,74)
(7,51)-(7,53)
(7,55)-(7,73)
(7,68)-(7,69)
(7,56)-(7,67)
(7,57)-(7,62)
(7,63)-(7,64)
(7,65)-(7,66)
(7,70)-(7,72)
(9,19)-(10,69)
(10,2)-(10,69)
(10,8)-(10,9)
(10,23)-(10,25)
(10,36)-(10,69)
(10,39)-(10,44)
(10,39)-(10,40)
(10,43)-(10,44)
(10,50)-(10,62)
(10,50)-(10,60)
(10,61)-(10,62)
(10,68)-(10,69)
(12,11)-(21,34)
(12,14)-(21,34)
(13,2)-(21,34)
(13,11)-(20,51)
(14,4)-(20,51)
(14,10)-(17,73)
(14,12)-(17,73)
(15,6)-(17,73)
(15,20)-(15,21)
(16,6)-(17,73)
(16,20)-(16,21)
(17,6)-(17,73)
(17,9)-(17,31)
(17,9)-(17,25)
(17,10)-(17,19)
(17,11)-(17,13)
(17,16)-(17,18)
(17,22)-(17,24)
(17,29)-(17,31)
(17,37)-(17,38)
(17,44)-(17,73)
(17,45)-(17,46)
(17,48)-(17,72)
(17,49)-(17,65)
(17,50)-(17,59)
(17,51)-(17,53)
(17,56)-(17,58)
(17,62)-(17,64)
(17,69)-(17,71)
(18,4)-(20,51)
(18,15)-(18,23)
(18,16)-(18,22)
(18,17)-(18,18)
(18,20)-(18,21)
(19,4)-(20,51)
(19,15)-(19,33)
(19,15)-(19,27)
(19,28)-(19,30)
(19,31)-(19,33)
(20,4)-(20,51)
(20,18)-(20,44)
(20,18)-(20,32)
(20,33)-(20,34)
(20,35)-(20,39)
(20,40)-(20,44)
(20,48)-(20,51)
(21,2)-(21,34)
(21,2)-(21,12)
(21,13)-(21,34)
(21,14)-(21,17)
(21,18)-(21,33)
(21,19)-(21,26)
(21,27)-(21,29)
(21,30)-(21,32)
*)
