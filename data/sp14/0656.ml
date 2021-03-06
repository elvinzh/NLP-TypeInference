
let rec clone x n =
  match n with | n when n <= 0 -> [] | _ -> x :: (clone x (n - 1));;

let rec padZero l1 l2 =
  if (List.length l1) > (List.length l2)
  then (l1, ((clone 0 ((List.length l1) - (List.length l2))) @ l2))
  else (((clone 0 ((List.length l2) - (List.length l1))) @ l1), l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x',x'') = x in
      let (c,s) = a in
      if (List.length s) = ((List.length l1) - 1)
      then (c, ((((c + x') + x'') / 10) :: (((c + x') + x'') mod 10) :: s))
      else ((((c + x') + x'') / 10), ((((c + x') + x'') mod 10) :: s)) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec mulByDigit i l =
  if i > 0 then bigAdd l (mulByDigit (i - 1) l) else [0];;

let bigMul l1 l2 =
  let f a x =
    let (l1',a') = x in
    match a with | [] -> (l1, a') | h::t -> bigAdd (mulByDigit (h l1') a') in
  let base = (l1, [0]) in
  let args = List.rev l2 in let (_,res) = List.fold_left f base args in res;;


(* fix

let rec clone x n =
  match n with | n when n <= 0 -> [] | _ -> x :: (clone x (n - 1));;

let rec padZero l1 l2 =
  if (List.length l1) > (List.length l2)
  then (l1, ((clone 0 ((List.length l1) - (List.length l2))) @ l2))
  else (((clone 0 ((List.length l2) - (List.length l1))) @ l1), l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x',x'') = x in
      let (c,s) = a in
      if (List.length s) = ((List.length l1) - 1)
      then (c, ((((c + x') + x'') / 10) :: (((c + x') + x'') mod 10) :: s))
      else ((((c + x') + x'') / 10), ((((c + x') + x'') mod 10) :: s)) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec mulByDigit i l =
  if i > 0 then bigAdd l (mulByDigit (i - 1) l) else [0];;

let bigMul l1 l2 =
  let f a x =
    match a with | (l1',a') -> (l1', (bigAdd (mulByDigit x l1') a')) in
  let base = (l1, []) in
  let args = List.rev l2 in let (_,res) = List.fold_left f base args in res;;

*)

(* changed spans
(31,4)-(32,74)
(31,19)-(31,20)
(32,4)-(32,74)
(32,26)-(32,28)
(32,30)-(32,32)
(32,44)-(32,74)
(32,63)-(32,70)
(32,64)-(32,65)
(33,18)-(33,21)
(33,19)-(33,20)
*)

(* type error slice
(27,16)-(27,22)
(27,16)-(27,47)
(30,2)-(34,75)
(30,8)-(32,74)
(30,10)-(32,74)
(31,4)-(32,74)
(32,4)-(32,74)
(32,10)-(32,11)
(32,25)-(32,33)
(32,44)-(32,50)
(32,44)-(32,74)
(34,42)-(34,56)
(34,42)-(34,68)
(34,57)-(34,58)
*)

(* all spans
(2,14)-(3,66)
(2,16)-(3,66)
(3,2)-(3,66)
(3,8)-(3,9)
(3,24)-(3,30)
(3,24)-(3,25)
(3,29)-(3,30)
(3,34)-(3,36)
(3,44)-(3,66)
(3,44)-(3,45)
(3,49)-(3,66)
(3,50)-(3,55)
(3,56)-(3,57)
(3,58)-(3,65)
(3,59)-(3,60)
(3,63)-(3,64)
(5,16)-(8,67)
(5,19)-(8,67)
(6,2)-(8,67)
(6,5)-(6,40)
(6,5)-(6,21)
(6,6)-(6,17)
(6,18)-(6,20)
(6,24)-(6,40)
(6,25)-(6,36)
(6,37)-(6,39)
(7,7)-(7,67)
(7,8)-(7,10)
(7,12)-(7,66)
(7,61)-(7,62)
(7,13)-(7,60)
(7,14)-(7,19)
(7,20)-(7,21)
(7,22)-(7,59)
(7,23)-(7,39)
(7,24)-(7,35)
(7,36)-(7,38)
(7,42)-(7,58)
(7,43)-(7,54)
(7,55)-(7,57)
(7,63)-(7,65)
(8,7)-(8,67)
(8,8)-(8,62)
(8,57)-(8,58)
(8,9)-(8,56)
(8,10)-(8,15)
(8,16)-(8,17)
(8,18)-(8,55)
(8,19)-(8,35)
(8,20)-(8,31)
(8,32)-(8,34)
(8,38)-(8,54)
(8,39)-(8,50)
(8,51)-(8,53)
(8,59)-(8,61)
(8,64)-(8,66)
(10,19)-(11,69)
(11,2)-(11,69)
(11,8)-(11,9)
(11,23)-(11,25)
(11,36)-(11,69)
(11,39)-(11,44)
(11,39)-(11,40)
(11,43)-(11,44)
(11,50)-(11,62)
(11,50)-(11,60)
(11,61)-(11,62)
(11,68)-(11,69)
(13,11)-(24,34)
(13,14)-(24,34)
(14,2)-(24,34)
(14,11)-(23,51)
(15,4)-(23,51)
(15,10)-(20,70)
(15,12)-(20,70)
(16,6)-(20,70)
(16,21)-(16,22)
(17,6)-(20,70)
(17,18)-(17,19)
(18,6)-(20,70)
(18,9)-(18,49)
(18,9)-(18,24)
(18,10)-(18,21)
(18,22)-(18,23)
(18,27)-(18,49)
(18,28)-(18,44)
(18,29)-(18,40)
(18,41)-(18,43)
(18,47)-(18,48)
(19,11)-(19,75)
(19,12)-(19,13)
(19,15)-(19,74)
(19,16)-(19,39)
(19,17)-(19,33)
(19,18)-(19,26)
(19,19)-(19,20)
(19,23)-(19,25)
(19,29)-(19,32)
(19,36)-(19,38)
(19,43)-(19,73)
(19,43)-(19,68)
(19,44)-(19,60)
(19,45)-(19,53)
(19,46)-(19,47)
(19,50)-(19,52)
(19,56)-(19,59)
(19,65)-(19,67)
(19,72)-(19,73)
(20,11)-(20,70)
(20,12)-(20,35)
(20,13)-(20,29)
(20,14)-(20,22)
(20,15)-(20,16)
(20,19)-(20,21)
(20,25)-(20,28)
(20,32)-(20,34)
(20,37)-(20,69)
(20,38)-(20,63)
(20,39)-(20,55)
(20,40)-(20,48)
(20,41)-(20,42)
(20,45)-(20,47)
(20,51)-(20,54)
(20,60)-(20,62)
(20,67)-(20,68)
(21,4)-(23,51)
(21,15)-(21,22)
(21,16)-(21,17)
(21,19)-(21,21)
(22,4)-(23,51)
(22,15)-(22,44)
(22,15)-(22,23)
(22,24)-(22,44)
(22,25)-(22,37)
(22,38)-(22,40)
(22,41)-(22,43)
(23,4)-(23,51)
(23,18)-(23,44)
(23,18)-(23,32)
(23,33)-(23,34)
(23,35)-(23,39)
(23,40)-(23,44)
(23,48)-(23,51)
(24,2)-(24,34)
(24,2)-(24,12)
(24,13)-(24,34)
(24,14)-(24,17)
(24,18)-(24,33)
(24,19)-(24,26)
(24,27)-(24,29)
(24,30)-(24,32)
(26,19)-(27,56)
(26,21)-(27,56)
(27,2)-(27,56)
(27,5)-(27,10)
(27,5)-(27,6)
(27,9)-(27,10)
(27,16)-(27,47)
(27,16)-(27,22)
(27,23)-(27,24)
(27,25)-(27,47)
(27,26)-(27,36)
(27,37)-(27,44)
(27,38)-(27,39)
(27,42)-(27,43)
(27,45)-(27,46)
(27,53)-(27,56)
(27,54)-(27,55)
(29,11)-(34,75)
(29,14)-(34,75)
(30,2)-(34,75)
(30,8)-(32,74)
(30,10)-(32,74)
(31,4)-(32,74)
(31,19)-(31,20)
(32,4)-(32,74)
(32,10)-(32,11)
(32,25)-(32,33)
(32,26)-(32,28)
(32,30)-(32,32)
(32,44)-(32,74)
(32,44)-(32,50)
(32,51)-(32,74)
(32,52)-(32,62)
(32,63)-(32,70)
(32,64)-(32,65)
(32,66)-(32,69)
(32,71)-(32,73)
(33,2)-(34,75)
(33,13)-(33,22)
(33,14)-(33,16)
(33,18)-(33,21)
(33,19)-(33,20)
(34,2)-(34,75)
(34,13)-(34,24)
(34,13)-(34,21)
(34,22)-(34,24)
(34,28)-(34,75)
(34,42)-(34,68)
(34,42)-(34,56)
(34,57)-(34,58)
(34,59)-(34,63)
(34,64)-(34,68)
(34,72)-(34,75)
*)
