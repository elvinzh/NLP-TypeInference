
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let diff = (List.length l1) - (List.length l2) in
  if diff >= 0
  then (l1, ((clone 0 diff) @ l2))
  else (((clone 0 (abs diff)) @ l1), l2);;

let rec removeZero l =
  match l with
  | [] -> []
  | h::t -> (match h with | 0 -> removeZero t | _ -> h :: t);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (arg1,arg2) = x in
      match a with
      | (0,_) ->
          if (arg1 + arg2) > 9
          then (1, [(arg1 + arg2) mod 10])
          else (0, [arg1 + arg2])
      | (_,_) ->
          if ((arg1 + arg2) + 1) > 9
          then (1, ([((arg1 + arg2) + 1) mod 10] @ a))
          else (0, (((arg1 + arg2) + 1) :: a)) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let diff = (List.length l1) - (List.length l2) in
  if diff >= 0
  then (l1, ((clone 0 diff) @ l2))
  else (((clone 0 (abs diff)) @ l1), l2);;

let rec removeZero l =
  match l with
  | [] -> []
  | h::t -> (match h with | 0 -> removeZero t | _ -> h :: t);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (arg1,arg2) = x in
      match a with
      | (0,acc) ->
          if (arg1 + arg2) > 9
          then (1, (((arg1 + arg2) mod 10) :: acc))
          else (0, ((arg1 + arg2) :: acc))
      | (0,[]) ->
          if (arg1 + arg2) > 9
          then (1, [(arg1 + arg2) mod 10])
          else (0, [arg1 + arg2])
      | (_,acc) ->
          if ((arg1 + arg2) + 1) > 9
          then (1, ((((arg1 + arg2) + 1) mod 10) :: acc))
          else (0, (((arg1 + arg2) + 1) :: acc)) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(19,6)-(27,46)
(22,19)-(22,41)
(26,19)-(26,53)
(26,20)-(26,48)
(26,49)-(26,50)
(26,51)-(26,52)
(27,43)-(27,44)
*)

(* type error slice
(17,4)-(30,51)
(17,10)-(27,46)
(17,12)-(27,46)
(18,6)-(27,46)
(19,6)-(27,46)
(19,12)-(19,13)
(21,10)-(23,33)
(23,15)-(23,33)
(26,19)-(26,53)
(26,49)-(26,50)
(26,51)-(26,52)
(30,18)-(30,32)
(30,18)-(30,44)
(30,33)-(30,34)
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
(4,12)-(8,40)
(4,15)-(8,40)
(5,2)-(8,40)
(5,13)-(5,48)
(5,13)-(5,29)
(5,14)-(5,25)
(5,26)-(5,28)
(5,32)-(5,48)
(5,33)-(5,44)
(5,45)-(5,47)
(6,2)-(8,40)
(6,5)-(6,14)
(6,5)-(6,9)
(6,13)-(6,14)
(7,7)-(7,34)
(7,8)-(7,10)
(7,12)-(7,33)
(7,28)-(7,29)
(7,13)-(7,27)
(7,14)-(7,19)
(7,20)-(7,21)
(7,22)-(7,26)
(7,30)-(7,32)
(8,7)-(8,40)
(8,8)-(8,35)
(8,30)-(8,31)
(8,9)-(8,29)
(8,10)-(8,15)
(8,16)-(8,17)
(8,18)-(8,28)
(8,19)-(8,22)
(8,23)-(8,27)
(8,32)-(8,34)
(8,37)-(8,39)
(10,19)-(13,60)
(11,2)-(13,60)
(11,8)-(11,9)
(12,10)-(12,12)
(13,12)-(13,60)
(13,19)-(13,20)
(13,33)-(13,45)
(13,33)-(13,43)
(13,44)-(13,45)
(13,53)-(13,59)
(13,53)-(13,54)
(13,58)-(13,59)
(15,11)-(31,34)
(15,14)-(31,34)
(16,2)-(31,34)
(16,11)-(30,51)
(17,4)-(30,51)
(17,10)-(27,46)
(17,12)-(27,46)
(18,6)-(27,46)
(18,24)-(18,25)
(19,6)-(27,46)
(19,12)-(19,13)
(21,10)-(23,33)
(21,13)-(21,30)
(21,13)-(21,26)
(21,14)-(21,18)
(21,21)-(21,25)
(21,29)-(21,30)
(22,15)-(22,42)
(22,16)-(22,17)
(22,19)-(22,41)
(22,20)-(22,40)
(22,20)-(22,33)
(22,21)-(22,25)
(22,28)-(22,32)
(22,38)-(22,40)
(23,15)-(23,33)
(23,16)-(23,17)
(23,19)-(23,32)
(23,20)-(23,31)
(23,20)-(23,24)
(23,27)-(23,31)
(25,10)-(27,46)
(25,13)-(25,36)
(25,13)-(25,32)
(25,14)-(25,27)
(25,15)-(25,19)
(25,22)-(25,26)
(25,30)-(25,31)
(25,35)-(25,36)
(26,15)-(26,54)
(26,16)-(26,17)
(26,19)-(26,53)
(26,49)-(26,50)
(26,20)-(26,48)
(26,21)-(26,47)
(26,21)-(26,40)
(26,22)-(26,35)
(26,23)-(26,27)
(26,30)-(26,34)
(26,38)-(26,39)
(26,45)-(26,47)
(26,51)-(26,52)
(27,15)-(27,46)
(27,16)-(27,17)
(27,19)-(27,45)
(27,20)-(27,39)
(27,21)-(27,34)
(27,22)-(27,26)
(27,29)-(27,33)
(27,37)-(27,38)
(27,43)-(27,44)
(28,4)-(30,51)
(28,15)-(28,22)
(28,16)-(28,17)
(28,19)-(28,21)
(29,4)-(30,51)
(29,15)-(29,44)
(29,15)-(29,23)
(29,24)-(29,44)
(29,25)-(29,37)
(29,38)-(29,40)
(29,41)-(29,43)
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
