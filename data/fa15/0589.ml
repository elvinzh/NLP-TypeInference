
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let rec exprToString e =
  match e with
  | VarX  -> "x"
  | VarY  -> "y"
  | Sine e1 -> "sin(pi*" ^ ((exprToString e1) ^ ")")
  | Cosine e1 -> "cos(pi*" ^ ((exprToString e1) ^ ")")
  | Average (e1,e2) ->
      "((" ^ ((exprToString e1) ^ (("+" exprToString e2) ^ ")/2)"))
  | Times (e1,e2) -> (exprToString e1) ^ ("*" ^ (exprToString e2))
  | Thresh (e1,e2,e3,e4) ->
      "(" ^
        ((exprToString e1) ^
           ("?" ^ ((exprToString e2) ^ (":" ^ ((exprToString e4) ^ ")")))));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let rec exprToString e =
  match e with
  | VarX  -> "x"
  | VarY  -> "y"
  | Sine e1 -> "sin(pi*" ^ ((exprToString e1) ^ ")")
  | Cosine e1 -> "cos(pi*" ^ ((exprToString e1) ^ ")")
  | Average (e1,e2) ->
      "((" ^ ((exprToString e1) ^ ("+" ^ ((exprToString e2) ^ ")/2)")))
  | Times (e1,e2) -> (exprToString e1) ^ ("*" ^ (exprToString e2))
  | Thresh (e1,e2,e3,e4) ->
      "(" ^
        ((exprToString e1) ^
           ("?" ^ ((exprToString e2) ^ (":" ^ ((exprToString e4) ^ ")")))));;

*)

(* changed spans
(18,35)-(18,56)
(18,40)-(18,52)
*)

(* type error slice
(18,35)-(18,56)
(18,36)-(18,39)
*)

(* all spans
(11,21)-(23,75)
(12,2)-(23,75)
(12,8)-(12,9)
(13,13)-(13,16)
(14,13)-(14,16)
(15,15)-(15,52)
(15,25)-(15,26)
(15,15)-(15,24)
(15,27)-(15,52)
(15,46)-(15,47)
(15,28)-(15,45)
(15,29)-(15,41)
(15,42)-(15,44)
(15,48)-(15,51)
(16,17)-(16,54)
(16,27)-(16,28)
(16,17)-(16,26)
(16,29)-(16,54)
(16,48)-(16,49)
(16,30)-(16,47)
(16,31)-(16,43)
(16,44)-(16,46)
(16,50)-(16,53)
(18,6)-(18,67)
(18,11)-(18,12)
(18,6)-(18,10)
(18,13)-(18,67)
(18,32)-(18,33)
(18,14)-(18,31)
(18,15)-(18,27)
(18,28)-(18,30)
(18,34)-(18,66)
(18,57)-(18,58)
(18,35)-(18,56)
(18,36)-(18,39)
(18,40)-(18,52)
(18,53)-(18,55)
(18,59)-(18,65)
(19,21)-(19,66)
(19,39)-(19,40)
(19,21)-(19,38)
(19,22)-(19,34)
(19,35)-(19,37)
(19,41)-(19,66)
(19,46)-(19,47)
(19,42)-(19,45)
(19,48)-(19,65)
(19,49)-(19,61)
(19,62)-(19,64)
(21,6)-(23,75)
(21,10)-(21,11)
(21,6)-(21,9)
(22,8)-(23,75)
(22,27)-(22,28)
(22,9)-(22,26)
(22,10)-(22,22)
(22,23)-(22,25)
(23,11)-(23,74)
(23,16)-(23,17)
(23,12)-(23,15)
(23,18)-(23,73)
(23,37)-(23,38)
(23,19)-(23,36)
(23,20)-(23,32)
(23,33)-(23,35)
(23,39)-(23,72)
(23,44)-(23,45)
(23,40)-(23,43)
(23,46)-(23,71)
(23,65)-(23,66)
(23,47)-(23,64)
(23,48)-(23,60)
(23,61)-(23,63)
(23,67)-(23,70)
*)
