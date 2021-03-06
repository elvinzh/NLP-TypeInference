
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Inverse of expr
  | Max of expr* expr
  | Range of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine a -> sin (pi *. (eval (a, x, y)))
  | Cosine a -> cos (pi *. (eval (a, x, y)))
  | Average (a,b) -> ((eval (a, x, y)) +. (eval (b, x, y))) /. 2.
  | Times (a,b) -> (eval (a, x, y)) *. (eval (b, x, y))
  | Thresh (a,b,c,d) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (c, x, y)
      else eval (d, x, y)
  | Inverse a ->
      let result = eval (a, x, y) in if result = 0. then 0 else 1 /. result
  | Max (a,b) ->
      let aResult = eval (a, x, y) in
      let bResult = eval (b, x, y) in
      if aResult > bResult then aResult else bResult
  | Range (a,b,c) ->
      let aResult = eval (a, x, y) in
      let bResult = eval (b, x, y) in
      let cResult = eval (c, x, y) in
      if aResult < bResult
      then bResult
      else if aResult < cResult then cResult else aResult;;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Inverse of expr
  | Max of expr* expr
  | Range of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine a -> sin (pi *. (eval (a, x, y)))
  | Cosine a -> cos (pi *. (eval (a, x, y)))
  | Average (a,b) -> ((eval (a, x, y)) +. (eval (b, x, y))) /. 2.
  | Times (a,b) -> (eval (a, x, y)) *. (eval (b, x, y))
  | Thresh (a,b,c,d) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (c, x, y)
      else eval (d, x, y)
  | Inverse a ->
      let result = eval (a, x, y) in if result = 0. then 0. else 1. /. result
  | Max (a,b) ->
      let aResult = eval (a, x, y) in
      let bResult = eval (b, x, y) in
      if aResult > bResult then aResult else bResult
  | Range (a,b,c) ->
      let aResult = eval (a, x, y) in
      let bResult = eval (b, x, y) in
      let cResult = eval (c, x, y) in
      if aResult < bResult
      then bResult
      else if aResult < cResult then cResult else aResult;;

*)

(* changed spans
(29,57)-(29,58)
(29,64)-(29,65)
*)

(* type error slice
(29,37)-(29,75)
(29,57)-(29,58)
(29,64)-(29,65)
(29,64)-(29,75)
*)

(* all spans
(14,9)-(14,26)
(14,9)-(14,12)
(14,16)-(14,26)
(14,17)-(14,21)
(14,22)-(14,25)
(16,14)-(40,57)
(17,2)-(40,57)
(17,8)-(17,9)
(18,13)-(18,14)
(19,13)-(19,14)
(20,14)-(20,42)
(20,14)-(20,17)
(20,18)-(20,42)
(20,19)-(20,21)
(20,25)-(20,41)
(20,26)-(20,30)
(20,31)-(20,40)
(20,32)-(20,33)
(20,35)-(20,36)
(20,38)-(20,39)
(21,16)-(21,44)
(21,16)-(21,19)
(21,20)-(21,44)
(21,21)-(21,23)
(21,27)-(21,43)
(21,28)-(21,32)
(21,33)-(21,42)
(21,34)-(21,35)
(21,37)-(21,38)
(21,40)-(21,41)
(22,21)-(22,65)
(22,21)-(22,59)
(22,22)-(22,38)
(22,23)-(22,27)
(22,28)-(22,37)
(22,29)-(22,30)
(22,32)-(22,33)
(22,35)-(22,36)
(22,42)-(22,58)
(22,43)-(22,47)
(22,48)-(22,57)
(22,49)-(22,50)
(22,52)-(22,53)
(22,55)-(22,56)
(22,63)-(22,65)
(23,19)-(23,55)
(23,19)-(23,35)
(23,20)-(23,24)
(23,25)-(23,34)
(23,26)-(23,27)
(23,29)-(23,30)
(23,32)-(23,33)
(23,39)-(23,55)
(23,40)-(23,44)
(23,45)-(23,54)
(23,46)-(23,47)
(23,49)-(23,50)
(23,52)-(23,53)
(25,6)-(27,25)
(25,9)-(25,44)
(25,9)-(25,25)
(25,10)-(25,14)
(25,15)-(25,24)
(25,16)-(25,17)
(25,19)-(25,20)
(25,22)-(25,23)
(25,28)-(25,44)
(25,29)-(25,33)
(25,34)-(25,43)
(25,35)-(25,36)
(25,38)-(25,39)
(25,41)-(25,42)
(26,11)-(26,25)
(26,11)-(26,15)
(26,16)-(26,25)
(26,17)-(26,18)
(26,20)-(26,21)
(26,23)-(26,24)
(27,11)-(27,25)
(27,11)-(27,15)
(27,16)-(27,25)
(27,17)-(27,18)
(27,20)-(27,21)
(27,23)-(27,24)
(29,6)-(29,75)
(29,19)-(29,33)
(29,19)-(29,23)
(29,24)-(29,33)
(29,25)-(29,26)
(29,28)-(29,29)
(29,31)-(29,32)
(29,37)-(29,75)
(29,40)-(29,51)
(29,40)-(29,46)
(29,49)-(29,51)
(29,57)-(29,58)
(29,64)-(29,75)
(29,64)-(29,65)
(29,69)-(29,75)
(31,6)-(33,52)
(31,20)-(31,34)
(31,20)-(31,24)
(31,25)-(31,34)
(31,26)-(31,27)
(31,29)-(31,30)
(31,32)-(31,33)
(32,6)-(33,52)
(32,20)-(32,34)
(32,20)-(32,24)
(32,25)-(32,34)
(32,26)-(32,27)
(32,29)-(32,30)
(32,32)-(32,33)
(33,6)-(33,52)
(33,9)-(33,26)
(33,9)-(33,16)
(33,19)-(33,26)
(33,32)-(33,39)
(33,45)-(33,52)
(35,6)-(40,57)
(35,20)-(35,34)
(35,20)-(35,24)
(35,25)-(35,34)
(35,26)-(35,27)
(35,29)-(35,30)
(35,32)-(35,33)
(36,6)-(40,57)
(36,20)-(36,34)
(36,20)-(36,24)
(36,25)-(36,34)
(36,26)-(36,27)
(36,29)-(36,30)
(36,32)-(36,33)
(37,6)-(40,57)
(37,20)-(37,34)
(37,20)-(37,24)
(37,25)-(37,34)
(37,26)-(37,27)
(37,29)-(37,30)
(37,32)-(37,33)
(38,6)-(40,57)
(38,9)-(38,26)
(38,9)-(38,16)
(38,19)-(38,26)
(39,11)-(39,18)
(40,11)-(40,57)
(40,14)-(40,31)
(40,14)-(40,21)
(40,24)-(40,31)
(40,37)-(40,44)
(40,50)-(40,57)
*)
