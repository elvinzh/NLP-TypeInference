
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  (((clone 0 ((List.length l2) - (List.length l1))) @ l1),
    ((clone 0 ((List.length l1) - (List.length l2))) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (num1,num2) = x in
      let (carry,sum) = a in
      ((((num1 + num2) + carry) / 10), ([((num1 + num2) + carry) mod 10] ::
        sum)) in
    let base = (0, 0) in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  (((clone 0 ((List.length l2) - (List.length l1))) @ l1),
    ((clone 0 ((List.length l1) - (List.length l2))) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (num1,num2) = x in
      let (carry,sum) = a in
      ((((num1 + num2) + carry) / 10), ((((num1 + num2) + carry) mod 10) ::
        sum)) in
    let base = (0, []) in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(16,40)-(16,72)
(18,19)-(18,20)
*)

(* type error slice
(13,4)-(20,51)
(13,10)-(17,13)
(15,6)-(17,13)
(15,24)-(15,25)
(16,39)-(17,12)
(17,8)-(17,11)
(18,4)-(20,51)
(18,15)-(18,21)
(18,19)-(18,20)
(20,18)-(20,32)
(20,18)-(20,44)
(20,33)-(20,34)
(20,35)-(20,39)
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
(4,12)-(6,59)
(4,15)-(6,59)
(5,2)-(6,59)
(5,3)-(5,57)
(5,52)-(5,53)
(5,4)-(5,51)
(5,5)-(5,10)
(5,11)-(5,12)
(5,13)-(5,50)
(5,14)-(5,30)
(5,15)-(5,26)
(5,27)-(5,29)
(5,33)-(5,49)
(5,34)-(5,45)
(5,46)-(5,48)
(5,54)-(5,56)
(6,4)-(6,58)
(6,53)-(6,54)
(6,5)-(6,52)
(6,6)-(6,11)
(6,12)-(6,13)
(6,14)-(6,51)
(6,15)-(6,31)
(6,16)-(6,27)
(6,28)-(6,30)
(6,34)-(6,50)
(6,35)-(6,46)
(6,47)-(6,49)
(6,55)-(6,57)
(8,19)-(9,74)
(9,2)-(9,74)
(9,8)-(9,9)
(9,23)-(9,25)
(9,36)-(9,74)
(9,39)-(9,44)
(9,39)-(9,40)
(9,43)-(9,44)
(9,50)-(9,62)
(9,50)-(9,60)
(9,61)-(9,62)
(9,68)-(9,74)
(9,68)-(9,69)
(9,73)-(9,74)
(11,11)-(21,34)
(11,14)-(21,34)
(12,2)-(21,34)
(12,11)-(20,51)
(13,4)-(20,51)
(13,10)-(17,13)
(13,12)-(17,13)
(14,6)-(17,13)
(14,24)-(14,25)
(15,6)-(17,13)
(15,24)-(15,25)
(16,6)-(17,13)
(16,7)-(16,37)
(16,8)-(16,31)
(16,9)-(16,22)
(16,10)-(16,14)
(16,17)-(16,21)
(16,25)-(16,30)
(16,34)-(16,36)
(16,39)-(17,12)
(16,40)-(16,72)
(16,41)-(16,71)
(16,41)-(16,64)
(16,42)-(16,55)
(16,43)-(16,47)
(16,50)-(16,54)
(16,58)-(16,63)
(16,69)-(16,71)
(17,8)-(17,11)
(18,4)-(20,51)
(18,15)-(18,21)
(18,16)-(18,17)
(18,19)-(18,20)
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
