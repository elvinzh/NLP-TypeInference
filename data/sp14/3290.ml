
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
      match a with
      | (lh1::lt1,lh2::lt2) ->
          (match x with
           | (h1,h2) ->
               (match h1 with
                | x::y ->
                    (match h2 with
                     | a::b ->
                         (((((x + a) + lh1) / 10) :: lt1),
                           ((((x + a) + lh1) mod 10) :: lt2))
                     | ([],[]) ->
                         (match x with
                          | (h1,h2) -> ([(h1 + h2) / 10], [(h1 + h2) mod 10]))))) in
    let base = ([], []) in
    let args = List.combine l1 l2 in
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
    let f a x = match x with | (v1,v2) -> ([v1], [v2]) in
    let base = ([], []) in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(15,6)-(27,81)
(15,12)-(15,13)
(17,10)-(27,81)
(17,17)-(17,18)
(19,15)-(27,80)
(19,22)-(19,24)
(21,20)-(27,79)
(21,27)-(21,29)
(23,25)-(24,61)
(23,26)-(23,57)
(23,27)-(23,49)
(23,28)-(23,43)
(23,29)-(23,36)
(23,30)-(23,31)
(23,34)-(23,35)
(23,39)-(23,42)
(23,46)-(23,48)
(23,53)-(23,56)
(24,27)-(24,60)
(24,28)-(24,52)
(24,29)-(24,44)
(24,30)-(24,37)
(24,31)-(24,32)
(24,35)-(24,36)
(24,40)-(24,43)
(24,49)-(24,51)
(24,56)-(24,59)
(26,25)-(27,78)
(27,41)-(27,50)
(27,41)-(27,55)
(27,42)-(27,44)
(27,47)-(27,49)
(27,53)-(27,55)
(27,59)-(27,68)
(27,59)-(27,75)
(27,60)-(27,62)
(27,65)-(27,67)
(27,73)-(27,75)
*)

(* type error slice
(21,20)-(27,79)
(23,29)-(23,36)
(23,30)-(23,31)
(26,25)-(27,78)
(26,32)-(26,33)
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
(12,11)-(31,34)
(12,14)-(31,34)
(13,2)-(31,34)
(13,11)-(30,51)
(14,4)-(30,51)
(14,10)-(27,81)
(14,12)-(27,81)
(15,6)-(27,81)
(15,12)-(15,13)
(17,10)-(27,81)
(17,17)-(17,18)
(19,15)-(27,80)
(19,22)-(19,24)
(21,20)-(27,79)
(21,27)-(21,29)
(23,25)-(24,61)
(23,26)-(23,57)
(23,27)-(23,49)
(23,28)-(23,43)
(23,29)-(23,36)
(23,30)-(23,31)
(23,34)-(23,35)
(23,39)-(23,42)
(23,46)-(23,48)
(23,53)-(23,56)
(24,27)-(24,60)
(24,28)-(24,52)
(24,29)-(24,44)
(24,30)-(24,37)
(24,31)-(24,32)
(24,35)-(24,36)
(24,40)-(24,43)
(24,49)-(24,51)
(24,56)-(24,59)
(26,25)-(27,78)
(26,32)-(26,33)
(27,39)-(27,77)
(27,40)-(27,56)
(27,41)-(27,55)
(27,41)-(27,50)
(27,42)-(27,44)
(27,47)-(27,49)
(27,53)-(27,55)
(27,58)-(27,76)
(27,59)-(27,75)
(27,59)-(27,68)
(27,60)-(27,62)
(27,65)-(27,67)
(27,73)-(27,75)
(28,4)-(30,51)
(28,15)-(28,23)
(28,16)-(28,18)
(28,20)-(28,22)
(29,4)-(30,51)
(29,15)-(29,33)
(29,15)-(29,27)
(29,28)-(29,30)
(29,31)-(29,33)
(30,4)-(30,51)
(30,18)-(30,44)
(30,18)-(30,32)
(30,33)-(30,34)
(30,35)-(30,39)
(30,40)-(30,44)
(30,48)-(30,51)
(31,2)-(31,34)
(31,2)-(31,12)
(31,13)-(31,34)
(31,14)-(31,17)
(31,18)-(31,33)
(31,19)-(31,26)
(31,27)-(31,29)
(31,30)-(31,32)
*)
