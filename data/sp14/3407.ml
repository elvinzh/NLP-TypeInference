
let rec clone x n =
  if n < 1
  then []
  else
    (let rec helper acc f x =
       match x with | 0 -> acc | _ -> helper (f :: acc) f (x - 1) in
     helper [] x n);;

let padZero l1 l2 =
  let x = (List.length l1) - (List.length l2) in
  if x != 0
  then
    (if x < 0
     then (((clone 0 (abs x)) @ l1), l2)
     else (l1, ((clone 0 (abs x)) @ l2)))
  else (l1, l2);;

let rec removeZero l =
  match l with | x::xs -> if x = 0 then removeZero xs else l | _ -> l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (b,c) ->
          let sum = b + c in
          if sum < 10
          then
            (match a with
             | (len,[]) -> (len, [sum])
             | (len,x'::xs') ->
                 if x' = (-1)
                 then
                   (if sum = 9
                    then (len, ((-1) :: 0 :: xs'))
                    else (len, ((sum + 1) :: xs')))
                 else (len, (sum :: x' :: xs')))
          else
            (match a with
             | (len,[]) -> (len, [(-1); sum mod 10])
             | (len,x'::xs') ->
                 if x' = (-1)
                 then (-1) :: ((sum mod 10) + 1) :: a
                 else (len, ((-1) :: (sum mod 10) :: x' :: xs'))) in
    let base = ((List.length l1), []) in
    let args = List.combine (List.rev l1) (List.rev l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n =
  if n < 1
  then []
  else
    (let rec helper acc f x =
       match x with | 0 -> acc | _ -> helper (f :: acc) f (x - 1) in
     helper [] x n);;

let padZero l1 l2 =
  let x = (List.length l1) - (List.length l2) in
  if x != 0
  then
    (if x < 0
     then (((clone 0 (abs x)) @ l1), l2)
     else (l1, ((clone 0 (abs x)) @ l2)))
  else (l1, l2);;

let rec removeZero l =
  match l with | x::xs -> if x = 0 then removeZero xs else l | _ -> l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (b,c) ->
          let sum = b + c in
          if sum < 10
          then
            (match a with
             | (len,[]) -> (len, [sum])
             | (len,x'::xs') ->
                 if x' = (-1)
                 then
                   (if sum = 9
                    then (len, ((-1) :: 0 :: xs'))
                    else (len, ((sum + 1) :: xs')))
                 else (len, (sum :: x' :: xs')))
          else
            (match a with
             | (len,[]) -> (len, [(-1); sum mod 10])
             | (len,x'::xs') ->
                 if x' = (-1)
                 then (len, ((-1) :: ((sum mod 10) + 1) :: xs'))
                 else (len, ((-1) :: (sum mod 10) :: x' :: xs'))) in
    let base = ((List.length l1), []) in
    let args = List.combine (List.rev l1) (List.rev l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(44,22)-(44,53)
(44,52)-(44,53)
*)

(* type error slice
(30,12)-(38,48)
(30,19)-(30,20)
(43,17)-(45,64)
(44,22)-(44,53)
(44,30)-(44,53)
(44,52)-(44,53)
(45,22)-(45,64)
*)

(* all spans
(2,14)-(8,19)
(2,16)-(8,19)
(3,2)-(8,19)
(3,5)-(3,10)
(3,5)-(3,6)
(3,9)-(3,10)
(4,7)-(4,9)
(6,4)-(8,19)
(6,20)-(7,65)
(6,24)-(7,65)
(6,26)-(7,65)
(7,7)-(7,65)
(7,13)-(7,14)
(7,27)-(7,30)
(7,38)-(7,65)
(7,38)-(7,44)
(7,45)-(7,55)
(7,46)-(7,47)
(7,51)-(7,54)
(7,56)-(7,57)
(7,58)-(7,65)
(7,59)-(7,60)
(7,63)-(7,64)
(8,5)-(8,18)
(8,5)-(8,11)
(8,12)-(8,14)
(8,15)-(8,16)
(8,17)-(8,18)
(10,12)-(17,15)
(10,15)-(17,15)
(11,2)-(17,15)
(11,10)-(11,45)
(11,10)-(11,26)
(11,11)-(11,22)
(11,23)-(11,25)
(11,29)-(11,45)
(11,30)-(11,41)
(11,42)-(11,44)
(12,2)-(17,15)
(12,5)-(12,11)
(12,5)-(12,6)
(12,10)-(12,11)
(14,4)-(16,41)
(14,8)-(14,13)
(14,8)-(14,9)
(14,12)-(14,13)
(15,10)-(15,40)
(15,11)-(15,35)
(15,30)-(15,31)
(15,12)-(15,29)
(15,13)-(15,18)
(15,19)-(15,20)
(15,21)-(15,28)
(15,22)-(15,25)
(15,26)-(15,27)
(15,32)-(15,34)
(15,37)-(15,39)
(16,10)-(16,40)
(16,11)-(16,13)
(16,15)-(16,39)
(16,34)-(16,35)
(16,16)-(16,33)
(16,17)-(16,22)
(16,23)-(16,24)
(16,25)-(16,32)
(16,26)-(16,29)
(16,30)-(16,31)
(16,36)-(16,38)
(17,7)-(17,15)
(17,8)-(17,10)
(17,12)-(17,14)
(19,19)-(20,69)
(20,2)-(20,69)
(20,8)-(20,9)
(20,26)-(20,60)
(20,29)-(20,34)
(20,29)-(20,30)
(20,33)-(20,34)
(20,40)-(20,53)
(20,40)-(20,50)
(20,51)-(20,53)
(20,59)-(20,60)
(20,68)-(20,69)
(22,11)-(49,34)
(22,14)-(49,34)
(23,2)-(49,34)
(23,11)-(48,51)
(24,4)-(48,51)
(24,10)-(45,65)
(24,12)-(45,65)
(25,6)-(45,65)
(25,12)-(25,13)
(27,10)-(45,65)
(27,20)-(27,25)
(27,20)-(27,21)
(27,24)-(27,25)
(28,10)-(45,65)
(28,13)-(28,21)
(28,13)-(28,16)
(28,19)-(28,21)
(30,12)-(38,48)
(30,19)-(30,20)
(31,27)-(31,39)
(31,28)-(31,31)
(31,33)-(31,38)
(31,34)-(31,37)
(33,17)-(38,47)
(33,20)-(33,29)
(33,20)-(33,22)
(33,25)-(33,29)
(35,19)-(37,51)
(35,23)-(35,30)
(35,23)-(35,26)
(35,29)-(35,30)
(36,25)-(36,50)
(36,26)-(36,29)
(36,31)-(36,49)
(36,32)-(36,36)
(36,40)-(36,48)
(36,40)-(36,41)
(36,45)-(36,48)
(37,25)-(37,50)
(37,26)-(37,29)
(37,31)-(37,49)
(37,32)-(37,41)
(37,33)-(37,36)
(37,39)-(37,40)
(37,45)-(37,48)
(38,22)-(38,47)
(38,23)-(38,26)
(38,28)-(38,46)
(38,29)-(38,32)
(38,36)-(38,45)
(38,36)-(38,38)
(38,42)-(38,45)
(40,12)-(45,65)
(40,19)-(40,20)
(41,27)-(41,52)
(41,28)-(41,31)
(41,33)-(41,51)
(41,34)-(41,38)
(41,40)-(41,50)
(41,40)-(41,43)
(41,48)-(41,50)
(43,17)-(45,64)
(43,20)-(43,29)
(43,20)-(43,22)
(43,25)-(43,29)
(44,22)-(44,53)
(44,22)-(44,26)
(44,30)-(44,53)
(44,30)-(44,48)
(44,31)-(44,43)
(44,32)-(44,35)
(44,40)-(44,42)
(44,46)-(44,47)
(44,52)-(44,53)
(45,22)-(45,64)
(45,23)-(45,26)
(45,28)-(45,63)
(45,29)-(45,33)
(45,37)-(45,62)
(45,37)-(45,49)
(45,38)-(45,41)
(45,46)-(45,48)
(45,53)-(45,62)
(45,53)-(45,55)
(45,59)-(45,62)
(46,4)-(48,51)
(46,15)-(46,37)
(46,16)-(46,32)
(46,17)-(46,28)
(46,29)-(46,31)
(46,34)-(46,36)
(47,4)-(48,51)
(47,15)-(47,55)
(47,15)-(47,27)
(47,28)-(47,41)
(47,29)-(47,37)
(47,38)-(47,40)
(47,42)-(47,55)
(47,43)-(47,51)
(47,52)-(47,54)
(48,4)-(48,51)
(48,18)-(48,44)
(48,18)-(48,32)
(48,33)-(48,34)
(48,35)-(48,39)
(48,40)-(48,44)
(48,48)-(48,51)
(49,2)-(49,34)
(49,2)-(49,12)
(49,13)-(49,34)
(49,14)-(49,17)
(49,18)-(49,33)
(49,19)-(49,26)
(49,27)-(49,29)
(49,30)-(49,32)
*)
