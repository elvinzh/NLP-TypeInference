
let rec clone x n = if n < 1 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  if (List.length l1) < (List.length l2)
  then (((clone 0 ((List.length l2) - (List.length l1))) @ l1), l2)
  else
    if (List.length l1) > (List.length l2)
    then (l1, ((clone 0 ((List.length l1) - (List.length l2))) @ l2))
    else (l1, l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let sum = (fst x) + (snd x) in
      match a with
      | h::t -> (h + (sum / 10)) :: ((h + sum) mod 10) :: t
      | _ -> (((fst x) + (snd x)) / 10) :: (((fst x) + (snd x)) mod 10) in
    let base = [] in
    let args = List.rev (List.combine l1 l2) in List.fold_left f base args in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n < 1 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  if (List.length l1) < (List.length l2)
  then (((clone 0 ((List.length l2) - (List.length l1))) @ l1), l2)
  else
    if (List.length l1) > (List.length l2)
    then (l1, ((clone 0 ((List.length l1) - (List.length l2))) @ l2))
    else (l1, l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let sum = (fst x) + (snd x) in
      match a with | h::t -> ((h + sum) / 10) :: ((h + sum) mod 10) :: t in
    let base = [] in
    let args = List.rev (List.combine l1 l2) in List.fold_left f base args in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(19,6)-(21,71)
(20,16)-(20,32)
(20,21)-(20,31)
(21,13)-(21,39)
(21,13)-(21,71)
(21,14)-(21,33)
(21,15)-(21,22)
(21,16)-(21,19)
(21,20)-(21,21)
(21,25)-(21,32)
(21,26)-(21,29)
(21,30)-(21,31)
(21,36)-(21,38)
(21,43)-(21,71)
(21,44)-(21,63)
(21,45)-(21,52)
(21,46)-(21,49)
(21,50)-(21,51)
(21,55)-(21,62)
(21,56)-(21,59)
(21,60)-(21,61)
(21,68)-(21,70)
*)

(* type error slice
(21,13)-(21,71)
(21,43)-(21,71)
*)

(* all spans
(2,14)-(2,64)
(2,16)-(2,64)
(2,20)-(2,64)
(2,23)-(2,28)
(2,23)-(2,24)
(2,27)-(2,28)
(2,34)-(2,36)
(2,42)-(2,64)
(2,42)-(2,43)
(2,47)-(2,64)
(2,48)-(2,53)
(2,54)-(2,55)
(2,56)-(2,63)
(2,57)-(2,58)
(2,61)-(2,62)
(4,12)-(10,17)
(4,15)-(10,17)
(5,2)-(10,17)
(5,5)-(5,40)
(5,5)-(5,21)
(5,6)-(5,17)
(5,18)-(5,20)
(5,24)-(5,40)
(5,25)-(5,36)
(5,37)-(5,39)
(6,7)-(6,67)
(6,8)-(6,62)
(6,57)-(6,58)
(6,9)-(6,56)
(6,10)-(6,15)
(6,16)-(6,17)
(6,18)-(6,55)
(6,19)-(6,35)
(6,20)-(6,31)
(6,32)-(6,34)
(6,38)-(6,54)
(6,39)-(6,50)
(6,51)-(6,53)
(6,59)-(6,61)
(6,64)-(6,66)
(8,4)-(10,17)
(8,7)-(8,42)
(8,7)-(8,23)
(8,8)-(8,19)
(8,20)-(8,22)
(8,26)-(8,42)
(8,27)-(8,38)
(8,39)-(8,41)
(9,9)-(9,69)
(9,10)-(9,12)
(9,14)-(9,68)
(9,63)-(9,64)
(9,15)-(9,62)
(9,16)-(9,21)
(9,22)-(9,23)
(9,24)-(9,61)
(9,25)-(9,41)
(9,26)-(9,37)
(9,38)-(9,40)
(9,44)-(9,60)
(9,45)-(9,56)
(9,57)-(9,59)
(9,65)-(9,67)
(10,9)-(10,17)
(10,10)-(10,12)
(10,14)-(10,16)
(12,19)-(13,69)
(13,2)-(13,69)
(13,8)-(13,9)
(13,23)-(13,25)
(13,36)-(13,69)
(13,39)-(13,44)
(13,39)-(13,40)
(13,43)-(13,44)
(13,50)-(13,62)
(13,50)-(13,60)
(13,61)-(13,62)
(13,68)-(13,69)
(15,11)-(24,34)
(15,14)-(24,34)
(16,2)-(24,34)
(16,11)-(23,74)
(17,4)-(23,74)
(17,10)-(21,71)
(17,12)-(21,71)
(18,6)-(21,71)
(18,16)-(18,33)
(18,16)-(18,23)
(18,17)-(18,20)
(18,21)-(18,22)
(18,26)-(18,33)
(18,27)-(18,30)
(18,31)-(18,32)
(19,6)-(21,71)
(19,12)-(19,13)
(20,16)-(20,59)
(20,16)-(20,32)
(20,17)-(20,18)
(20,21)-(20,31)
(20,22)-(20,25)
(20,28)-(20,30)
(20,36)-(20,59)
(20,36)-(20,54)
(20,37)-(20,46)
(20,38)-(20,39)
(20,42)-(20,45)
(20,51)-(20,53)
(20,58)-(20,59)
(21,13)-(21,71)
(21,13)-(21,39)
(21,14)-(21,33)
(21,15)-(21,22)
(21,16)-(21,19)
(21,20)-(21,21)
(21,25)-(21,32)
(21,26)-(21,29)
(21,30)-(21,31)
(21,36)-(21,38)
(21,43)-(21,71)
(21,44)-(21,63)
(21,45)-(21,52)
(21,46)-(21,49)
(21,50)-(21,51)
(21,55)-(21,62)
(21,56)-(21,59)
(21,60)-(21,61)
(21,68)-(21,70)
(22,4)-(23,74)
(22,15)-(22,17)
(23,4)-(23,74)
(23,15)-(23,44)
(23,15)-(23,23)
(23,24)-(23,44)
(23,25)-(23,37)
(23,38)-(23,40)
(23,41)-(23,43)
(23,48)-(23,74)
(23,48)-(23,62)
(23,63)-(23,64)
(23,65)-(23,69)
(23,70)-(23,74)
(24,2)-(24,34)
(24,2)-(24,12)
(24,13)-(24,34)
(24,14)-(24,17)
(24,18)-(24,33)
(24,19)-(24,26)
(24,27)-(24,29)
(24,30)-(24,32)
*)
