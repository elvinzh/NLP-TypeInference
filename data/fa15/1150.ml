
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
      let (i,j) = x in
      match a with
      | (c,d) ->
          if ((i + j) + c) > 9
          then (1, ((((i + j) + c) mod 10) :: d))
          else (0, ((((i + j) + c) mod 10) :: d)) in
    let base = (0, []) in
    let args = (List.rev (List.combine l1 l2)) @ [(0, 0)] in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = a ^ (sep ^ x) in
      let base = h in let l = t in List.fold_left f base l;;

let carryFunc p = let z = List.rev p in match z with | h::t -> List.rev t;;

let intListToInt l = int_of_string (sepConcat "" (List.map string_of_int l));;

let rec mulByDigit i l =
  if i > 0 then bigAdd l (mulByDigit (i - 1) l) else [];;

let bigMul l1 l2 =
  let f a x =
    match a with
    | (r,v) ->
        let sum = intListToInt (mulByDigit (intListToInt l1) x) in
        if (sum + r) > 9
        then
          ((intListToInt (carryFunc (mulByDigit (intListToInt l1) x))),
            (((sum + r) mod 10) :: v))
        else (0, (((sum + r) mod 10) :: v)) in
  let base = (0, []) in
  let args = [0; 0; 0; 0; 0; 0; 0] @ l2 in
  let (_,res) = List.fold_left f base args in res;;


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
      let (i,j) = x in
      match a with
      | (c,d) ->
          if ((i + j) + c) > 9
          then (1, ((((i + j) + c) mod 10) :: d))
          else (0, ((((i + j) + c) mod 10) :: d)) in
    let base = (0, []) in
    let args = (List.rev (List.combine l1 l2)) @ [(0, 0)] in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = a ^ (sep ^ x) in
      let base = h in let l = t in List.fold_left f base l;;

let carryFunc p = let z = List.rev p in match z with | h::t -> List.rev t;;

let intListToInt l = int_of_string (sepConcat "" (List.map string_of_int l));;

let rec mulByDigit i l =
  if i > 0 then bigAdd l (mulByDigit (i - 1) l) else [];;

let bigMul l1 l2 =
  let f a x =
    match a with
    | (r,v) ->
        let sum = intListToInt (mulByDigit (intListToInt l1) [x]) in
        if (sum + r) > 9
        then
          ((intListToInt (carryFunc (mulByDigit (intListToInt l1) [x]))),
            (((sum + r) mod 10) :: v))
        else (0, (((sum + r) mod 10) :: v)) in
  let base = (0, []) in
  let args = l2 in let (_,res) = List.fold_left f base args in res;;

*)

(* changed spans
(47,61)-(47,62)
(50,66)-(50,67)
(54,13)-(54,34)
(54,13)-(54,39)
(54,14)-(54,15)
(54,17)-(54,18)
(54,20)-(54,21)
(54,23)-(54,24)
(54,26)-(54,27)
(54,29)-(54,30)
(54,32)-(54,33)
(54,35)-(54,36)
*)

(* type error slice
(4,3)-(10,19)
(4,12)-(10,17)
(6,38)-(6,54)
(6,39)-(6,50)
(6,51)-(6,53)
(15,3)-(27,36)
(15,11)-(27,34)
(27,18)-(27,33)
(27,19)-(27,26)
(27,27)-(27,29)
(41,16)-(41,22)
(41,16)-(41,47)
(41,23)-(41,24)
(41,25)-(41,47)
(41,26)-(41,36)
(41,45)-(41,46)
(44,2)-(55,49)
(44,8)-(52,43)
(44,10)-(52,43)
(47,31)-(47,63)
(47,32)-(47,42)
(47,61)-(47,62)
(54,2)-(55,49)
(54,13)-(54,34)
(54,13)-(54,39)
(54,14)-(54,15)
(54,35)-(54,36)
(55,16)-(55,30)
(55,16)-(55,42)
(55,31)-(55,32)
(55,38)-(55,42)
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
(15,11)-(27,34)
(15,14)-(27,34)
(16,2)-(27,34)
(16,11)-(26,51)
(17,4)-(26,51)
(17,10)-(23,49)
(17,12)-(23,49)
(18,6)-(23,49)
(18,18)-(18,19)
(19,6)-(23,49)
(19,12)-(19,13)
(21,10)-(23,49)
(21,13)-(21,30)
(21,13)-(21,26)
(21,14)-(21,21)
(21,15)-(21,16)
(21,19)-(21,20)
(21,24)-(21,25)
(21,29)-(21,30)
(22,15)-(22,49)
(22,16)-(22,17)
(22,19)-(22,48)
(22,20)-(22,42)
(22,21)-(22,34)
(22,22)-(22,29)
(22,23)-(22,24)
(22,27)-(22,28)
(22,32)-(22,33)
(22,39)-(22,41)
(22,46)-(22,47)
(23,15)-(23,49)
(23,16)-(23,17)
(23,19)-(23,48)
(23,20)-(23,42)
(23,21)-(23,34)
(23,22)-(23,29)
(23,23)-(23,24)
(23,27)-(23,28)
(23,32)-(23,33)
(23,39)-(23,41)
(23,46)-(23,47)
(24,4)-(26,51)
(24,15)-(24,22)
(24,16)-(24,17)
(24,19)-(24,21)
(25,4)-(26,51)
(25,15)-(25,57)
(25,47)-(25,48)
(25,15)-(25,46)
(25,16)-(25,24)
(25,25)-(25,45)
(25,26)-(25,38)
(25,39)-(25,41)
(25,42)-(25,44)
(25,49)-(25,57)
(25,50)-(25,56)
(25,51)-(25,52)
(25,54)-(25,55)
(26,4)-(26,51)
(26,18)-(26,44)
(26,18)-(26,32)
(26,33)-(26,34)
(26,35)-(26,39)
(26,40)-(26,44)
(26,48)-(26,51)
(27,2)-(27,34)
(27,2)-(27,12)
(27,13)-(27,34)
(27,14)-(27,17)
(27,18)-(27,33)
(27,19)-(27,26)
(27,27)-(27,29)
(27,30)-(27,32)
(29,18)-(34,58)
(29,22)-(34,58)
(30,2)-(34,58)
(30,8)-(30,10)
(31,10)-(31,12)
(33,6)-(34,58)
(33,12)-(33,31)
(33,14)-(33,31)
(33,18)-(33,31)
(33,20)-(33,21)
(33,18)-(33,19)
(33,22)-(33,31)
(33,27)-(33,28)
(33,23)-(33,26)
(33,29)-(33,30)
(34,6)-(34,58)
(34,17)-(34,18)
(34,22)-(34,58)
(34,30)-(34,31)
(34,35)-(34,58)
(34,35)-(34,49)
(34,50)-(34,51)
(34,52)-(34,56)
(34,57)-(34,58)
(36,14)-(36,73)
(36,18)-(36,73)
(36,26)-(36,36)
(36,26)-(36,34)
(36,35)-(36,36)
(36,40)-(36,73)
(36,46)-(36,47)
(36,63)-(36,73)
(36,63)-(36,71)
(36,72)-(36,73)
(38,17)-(38,76)
(38,21)-(38,76)
(38,21)-(38,34)
(38,35)-(38,76)
(38,36)-(38,45)
(38,46)-(38,48)
(38,49)-(38,75)
(38,50)-(38,58)
(38,59)-(38,72)
(38,73)-(38,74)
(40,19)-(41,55)
(40,21)-(41,55)
(41,2)-(41,55)
(41,5)-(41,10)
(41,5)-(41,6)
(41,9)-(41,10)
(41,16)-(41,47)
(41,16)-(41,22)
(41,23)-(41,24)
(41,25)-(41,47)
(41,26)-(41,36)
(41,37)-(41,44)
(41,38)-(41,39)
(41,42)-(41,43)
(41,45)-(41,46)
(41,53)-(41,55)
(43,11)-(55,49)
(43,14)-(55,49)
(44,2)-(55,49)
(44,8)-(52,43)
(44,10)-(52,43)
(45,4)-(52,43)
(45,10)-(45,11)
(47,8)-(52,43)
(47,18)-(47,63)
(47,18)-(47,30)
(47,31)-(47,63)
(47,32)-(47,42)
(47,43)-(47,60)
(47,44)-(47,56)
(47,57)-(47,59)
(47,61)-(47,62)
(48,8)-(52,43)
(48,11)-(48,24)
(48,11)-(48,20)
(48,12)-(48,15)
(48,18)-(48,19)
(48,23)-(48,24)
(50,10)-(51,38)
(50,11)-(50,70)
(50,12)-(50,24)
(50,25)-(50,69)
(50,26)-(50,35)
(50,36)-(50,68)
(50,37)-(50,47)
(50,48)-(50,65)
(50,49)-(50,61)
(50,62)-(50,64)
(50,66)-(50,67)
(51,12)-(51,37)
(51,13)-(51,31)
(51,14)-(51,23)
(51,15)-(51,18)
(51,21)-(51,22)
(51,28)-(51,30)
(51,35)-(51,36)
(52,13)-(52,43)
(52,14)-(52,15)
(52,17)-(52,42)
(52,18)-(52,36)
(52,19)-(52,28)
(52,20)-(52,23)
(52,26)-(52,27)
(52,33)-(52,35)
(52,40)-(52,41)
(53,2)-(55,49)
(53,13)-(53,20)
(53,14)-(53,15)
(53,17)-(53,19)
(54,2)-(55,49)
(54,13)-(54,39)
(54,35)-(54,36)
(54,13)-(54,34)
(54,14)-(54,15)
(54,17)-(54,18)
(54,20)-(54,21)
(54,23)-(54,24)
(54,26)-(54,27)
(54,29)-(54,30)
(54,32)-(54,33)
(54,37)-(54,39)
(55,2)-(55,49)
(55,16)-(55,42)
(55,16)-(55,30)
(55,31)-(55,32)
(55,33)-(55,37)
(55,38)-(55,42)
(55,46)-(55,49)
*)
