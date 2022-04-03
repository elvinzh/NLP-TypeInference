
let rec clone x n = if n > 0 then x :: (clone x (n - 1)) else [];;

let padZero l1 l2 =
  if (List.length l1) < (List.length l2)
  then ((List.append (clone 0 ((List.length l2) - (List.length l1))) l1), l2)
  else (l1, (List.append (clone 0 ((List.length l1) - (List.length l2))) l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (v1,v2) ->
          (match a with
           | (list1,list2) ->
               (match list1 with
                | [] ->
                    ((((v1 + v2) / 10) :: list1), (((v1 + v2) mod 10) ::
                      list2))
                | h::t ->
                    (((((v1 + v2) + h) / 10) :: list1),
                      ((((v1 + v2) + h) mod 10) :: list2)))) in
    let base = ([], []) in
    let args = (List.rev (List.combine l1 l2)) :: (0, 0) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n > 0 then x :: (clone x (n - 1)) else [];;

let padZero l1 l2 =
  if (List.length l1) < (List.length l2)
  then ((List.append (clone 0 ((List.length l2) - (List.length l1))) l1), l2)
  else (l1, (List.append (clone 0 ((List.length l1) - (List.length l2))) l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (v1,v2) ->
          (match a with
           | (list1,list2) ->
               (match list1 with
                | [] ->
                    ((((v1 + v2) / 10) :: list1), (((v1 + v2) mod 10) ::
                      list2))
                | h::t ->
                    (((((v1 + v2) + h) / 10) :: list1),
                      ((((v1 + v2) + h) mod 10) :: list2)))) in
    let base = ([], []) in
    let args = List.append (List.rev (List.combine l1 l2)) [(0, 0)] in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(27,15)-(27,46)
(27,15)-(27,56)
(27,50)-(27,56)
*)

(* type error slice
(14,4)-(28,51)
(14,10)-(25,60)
(14,12)-(25,60)
(15,6)-(25,60)
(15,12)-(15,13)
(27,4)-(28,51)
(27,15)-(27,46)
(27,15)-(27,56)
(27,16)-(27,24)
(27,50)-(27,56)
(28,18)-(28,32)
(28,18)-(28,44)
(28,33)-(28,34)
(28,40)-(28,44)
*)

(* all spans
(2,14)-(2,64)
(2,16)-(2,64)
(2,20)-(2,64)
(2,23)-(2,28)
(2,23)-(2,24)
(2,27)-(2,28)
(2,34)-(2,56)
(2,34)-(2,35)
(2,39)-(2,56)
(2,40)-(2,45)
(2,46)-(2,47)
(2,48)-(2,55)
(2,49)-(2,50)
(2,53)-(2,54)
(2,62)-(2,64)
(4,12)-(7,77)
(4,15)-(7,77)
(5,2)-(7,77)
(5,5)-(5,40)
(5,5)-(5,21)
(5,6)-(5,17)
(5,18)-(5,20)
(5,24)-(5,40)
(5,25)-(5,36)
(5,37)-(5,39)
(6,7)-(6,77)
(6,8)-(6,72)
(6,9)-(6,20)
(6,21)-(6,68)
(6,22)-(6,27)
(6,28)-(6,29)
(6,30)-(6,67)
(6,31)-(6,47)
(6,32)-(6,43)
(6,44)-(6,46)
(6,50)-(6,66)
(6,51)-(6,62)
(6,63)-(6,65)
(6,69)-(6,71)
(6,74)-(6,76)
(7,7)-(7,77)
(7,8)-(7,10)
(7,12)-(7,76)
(7,13)-(7,24)
(7,25)-(7,72)
(7,26)-(7,31)
(7,32)-(7,33)
(7,34)-(7,71)
(7,35)-(7,51)
(7,36)-(7,47)
(7,48)-(7,50)
(7,54)-(7,70)
(7,55)-(7,66)
(7,67)-(7,69)
(7,73)-(7,75)
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
(12,11)-(29,34)
(12,14)-(29,34)
(13,2)-(29,34)
(13,11)-(28,51)
(14,4)-(28,51)
(14,10)-(25,60)
(14,12)-(25,60)
(15,6)-(25,60)
(15,12)-(15,13)
(17,10)-(25,60)
(17,17)-(17,18)
(19,15)-(25,59)
(19,22)-(19,27)
(21,20)-(22,29)
(21,21)-(21,48)
(21,22)-(21,38)
(21,23)-(21,32)
(21,24)-(21,26)
(21,29)-(21,31)
(21,35)-(21,37)
(21,42)-(21,47)
(21,50)-(22,28)
(21,51)-(21,69)
(21,52)-(21,61)
(21,53)-(21,55)
(21,58)-(21,60)
(21,66)-(21,68)
(22,22)-(22,27)
(24,20)-(25,58)
(24,21)-(24,54)
(24,22)-(24,44)
(24,23)-(24,38)
(24,24)-(24,33)
(24,25)-(24,27)
(24,30)-(24,32)
(24,36)-(24,37)
(24,41)-(24,43)
(24,48)-(24,53)
(25,22)-(25,57)
(25,23)-(25,47)
(25,24)-(25,39)
(25,25)-(25,34)
(25,26)-(25,28)
(25,31)-(25,33)
(25,37)-(25,38)
(25,44)-(25,46)
(25,51)-(25,56)
(26,4)-(28,51)
(26,15)-(26,23)
(26,16)-(26,18)
(26,20)-(26,22)
(27,4)-(28,51)
(27,15)-(27,56)
(27,15)-(27,46)
(27,16)-(27,24)
(27,25)-(27,45)
(27,26)-(27,38)
(27,39)-(27,41)
(27,42)-(27,44)
(27,50)-(27,56)
(27,51)-(27,52)
(27,54)-(27,55)
(28,4)-(28,51)
(28,18)-(28,44)
(28,18)-(28,32)
(28,33)-(28,34)
(28,35)-(28,39)
(28,40)-(28,44)
(28,48)-(28,51)
(29,2)-(29,34)
(29,2)-(29,12)
(29,13)-(29,34)
(29,14)-(29,17)
(29,18)-(29,33)
(29,19)-(29,26)
(29,27)-(29,29)
(29,30)-(29,32)
*)
