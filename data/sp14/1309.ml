
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let rec padZero l1 l2 =
  let diffsize = (List.length l1) - (List.length l2) in
  if diffsize > 0
  then (l1, (List.append (clone 0 diffsize) l2))
  else ((List.append (clone 0 ((-1) * diffsize)) l1), l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (h1,h2) -> (removeZero (((h1 + h2) / 10) :: ((h1 + h2) mod 10))) :: a
      | _ -> a in
    let base = [] in
    let args = List.combine l1 l2 in
    let res = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let rec padZero l1 l2 =
  let diffsize = (List.length l1) - (List.length l2) in
  if diffsize > 0
  then (l1, (List.append (clone 0 diffsize) l2))
  else ((List.append (clone 0 ((-1) * diffsize)) l1), l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (h1,h2) -> ((h1 + h2) / 10) :: ((h1 + h2) mod 10) :: a
      | _ -> a in
    let base = [] in
    let args = List.combine l1 l2 in
    let res = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(17,19)-(17,72)
(17,20)-(17,30)
(17,31)-(17,71)
(17,52)-(17,70)
*)

(* type error slice
(11,2)-(11,74)
(11,36)-(11,74)
(11,50)-(11,60)
(11,50)-(11,62)
(11,61)-(11,62)
(11,68)-(11,69)
(11,68)-(11,74)
(14,2)-(22,34)
(14,11)-(21,47)
(15,4)-(21,47)
(15,10)-(18,14)
(17,19)-(17,72)
(17,19)-(17,77)
(17,20)-(17,30)
(17,31)-(17,71)
(17,52)-(17,70)
(17,76)-(17,77)
(19,4)-(21,47)
(20,4)-(21,47)
(21,4)-(21,47)
(21,14)-(21,28)
(21,14)-(21,40)
(21,29)-(21,30)
(21,44)-(21,47)
(22,2)-(22,12)
(22,2)-(22,34)
(22,13)-(22,34)
(22,14)-(22,17)
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
(4,16)-(8,57)
(4,19)-(8,57)
(5,2)-(8,57)
(5,17)-(5,52)
(5,17)-(5,33)
(5,18)-(5,29)
(5,30)-(5,32)
(5,36)-(5,52)
(5,37)-(5,48)
(5,49)-(5,51)
(6,2)-(8,57)
(6,5)-(6,17)
(6,5)-(6,13)
(6,16)-(6,17)
(7,7)-(7,48)
(7,8)-(7,10)
(7,12)-(7,47)
(7,13)-(7,24)
(7,25)-(7,43)
(7,26)-(7,31)
(7,32)-(7,33)
(7,34)-(7,42)
(7,44)-(7,46)
(8,7)-(8,57)
(8,8)-(8,52)
(8,9)-(8,20)
(8,21)-(8,48)
(8,22)-(8,27)
(8,28)-(8,29)
(8,30)-(8,47)
(8,31)-(8,35)
(8,38)-(8,46)
(8,49)-(8,51)
(8,54)-(8,56)
(10,19)-(11,74)
(11,2)-(11,74)
(11,8)-(11,9)
(11,23)-(11,25)
(11,36)-(11,74)
(11,39)-(11,44)
(11,39)-(11,40)
(11,43)-(11,44)
(11,50)-(11,62)
(11,50)-(11,60)
(11,61)-(11,62)
(11,68)-(11,74)
(11,68)-(11,69)
(11,73)-(11,74)
(13,11)-(22,34)
(13,14)-(22,34)
(14,2)-(22,34)
(14,11)-(21,47)
(15,4)-(21,47)
(15,10)-(18,14)
(15,12)-(18,14)
(16,6)-(18,14)
(16,12)-(16,13)
(17,19)-(17,77)
(17,19)-(17,72)
(17,20)-(17,30)
(17,31)-(17,71)
(17,32)-(17,48)
(17,33)-(17,42)
(17,34)-(17,36)
(17,39)-(17,41)
(17,45)-(17,47)
(17,52)-(17,70)
(17,53)-(17,62)
(17,54)-(17,56)
(17,59)-(17,61)
(17,67)-(17,69)
(17,76)-(17,77)
(18,13)-(18,14)
(19,4)-(21,47)
(19,15)-(19,17)
(20,4)-(21,47)
(20,15)-(20,33)
(20,15)-(20,27)
(20,28)-(20,30)
(20,31)-(20,33)
(21,4)-(21,47)
(21,14)-(21,40)
(21,14)-(21,28)
(21,29)-(21,30)
(21,31)-(21,35)
(21,36)-(21,40)
(21,44)-(21,47)
(22,2)-(22,34)
(22,2)-(22,12)
(22,13)-(22,34)
(22,14)-(22,17)
(22,18)-(22,33)
(22,19)-(22,26)
(22,27)-(22,29)
(22,30)-(22,32)
*)
