
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
      match a with
      | (c,d) ->
          if (((fst x) + (snd x)) + c) > 9
          then (1, ((((fst x) + (snd x)) + c) mod 10))
          else (0, ((((fst x) + (snd x)) + c) mod 10)) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
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
      match a with
      | (c,h::t) ->
          if (((fst x) + (snd x)) + c) > 9
          then (1, (((((fst x) + (snd x)) + c) mod 10) :: t))
          else (0, (((((fst x) + (snd x)) + c) mod 10) :: t)) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(18,6)-(22,54)
(21,19)-(21,53)
(22,15)-(22,54)
(22,19)-(22,53)
(23,4)-(25,51)
*)

(* type error slice
(17,4)-(25,51)
(17,10)-(22,54)
(17,12)-(22,54)
(18,6)-(22,54)
(20,10)-(22,54)
(22,15)-(22,54)
(22,19)-(22,53)
(23,4)-(25,51)
(23,15)-(23,22)
(23,19)-(23,21)
(25,18)-(25,32)
(25,18)-(25,44)
(25,33)-(25,34)
(25,35)-(25,39)
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
(15,11)-(26,34)
(15,14)-(26,34)
(16,2)-(26,34)
(16,11)-(25,51)
(17,4)-(25,51)
(17,10)-(22,54)
(17,12)-(22,54)
(18,6)-(22,54)
(18,12)-(18,13)
(20,10)-(22,54)
(20,13)-(20,42)
(20,13)-(20,38)
(20,14)-(20,33)
(20,15)-(20,22)
(20,16)-(20,19)
(20,20)-(20,21)
(20,25)-(20,32)
(20,26)-(20,29)
(20,30)-(20,31)
(20,36)-(20,37)
(20,41)-(20,42)
(21,15)-(21,54)
(21,16)-(21,17)
(21,19)-(21,53)
(21,20)-(21,45)
(21,21)-(21,40)
(21,22)-(21,29)
(21,23)-(21,26)
(21,27)-(21,28)
(21,32)-(21,39)
(21,33)-(21,36)
(21,37)-(21,38)
(21,43)-(21,44)
(21,50)-(21,52)
(22,15)-(22,54)
(22,16)-(22,17)
(22,19)-(22,53)
(22,20)-(22,45)
(22,21)-(22,40)
(22,22)-(22,29)
(22,23)-(22,26)
(22,27)-(22,28)
(22,32)-(22,39)
(22,33)-(22,36)
(22,37)-(22,38)
(22,43)-(22,44)
(22,50)-(22,52)
(23,4)-(25,51)
(23,15)-(23,22)
(23,16)-(23,17)
(23,19)-(23,21)
(24,4)-(25,51)
(24,15)-(24,44)
(24,15)-(24,23)
(24,24)-(24,44)
(24,25)-(24,37)
(24,38)-(24,40)
(24,41)-(24,43)
(25,4)-(25,51)
(25,18)-(25,44)
(25,18)-(25,32)
(25,33)-(25,34)
(25,35)-(25,39)
(25,40)-(25,44)
(25,48)-(25,51)
(26,2)-(26,34)
(26,2)-(26,12)
(26,13)-(26,34)
(26,14)-(26,17)
(26,18)-(26,33)
(26,19)-(26,26)
(26,27)-(26,29)
(26,30)-(26,32)
*)