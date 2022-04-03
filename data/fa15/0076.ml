
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Uncreative of expr* expr* expr
  | Creative of expr;;

let rec exprToString e =
  match e with
  | VarX  -> "x"
  | VarY  -> "y"
  | Sine e -> "sin(pi*" ^ ((exprToString e) ^ ")")
  | Cosine e -> "cos(pi*" ^ ((exprToString e) ^ ")")
  | Average (e1,e2) ->
      "((" ^ ((exprToString e1) ^ ("+" ^ ((exprToString e2) ^ ")/2)")))
  | Times (e1,e2) -> (exprToString e1) ^ ("*" ^ (exprToString e2))
  | Thresh (e1,e2,e3,e4) ->
      "(" ^
        ((exprToString e1) ^
           ("<" ^
              ((exprToString e2) ^
                 ("?" ^
                    ((exprToString e3) ^ (":" ^ ((exprToString e4) ^ ")")))))))
  | Uncreative (e1,e2,e3) ->
      "(" ^
        ((exprToString e1) ^
           ("/2*" ^ ((exprToString e2 "/3*") ^ (exprToString e3 "/4)"))))
  | Creative e1 -> "(-1*" ^ ((exprToString e1) ^ ")");;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Uncreative of expr* expr* expr
  | Creative of expr;;

let rec exprToString e =
  match e with
  | VarX  -> "x"
  | VarY  -> "y"
  | Sine e -> "sin(pi*" ^ ((exprToString e) ^ ")")
  | Cosine e -> "cos(pi*" ^ ((exprToString e) ^ ")")
  | Average (e1,e2) ->
      "((" ^ ((exprToString e1) ^ ("+" ^ ((exprToString e2) ^ ")/2)")))
  | Times (e1,e2) -> (exprToString e1) ^ ("*" ^ (exprToString e2))
  | Thresh (e1,e2,e3,e4) ->
      "(" ^
        ((exprToString e1) ^
           ("<" ^
              ((exprToString e2) ^
                 ("?" ^
                    ((exprToString e3) ^ (":" ^ ((exprToString e4) ^ ")")))))))
  | Uncreative (e1,e2,e3) ->
      "(" ^
        ((exprToString e1) ^
           ("/2*" ^
              ((exprToString e2) ^ ("/3*" ^ ((exprToString e3) ^ "/4)")))))
  | Creative e1 -> "(-1*" ^ ((exprToString e1) ^ ")");;

*)

(* changed spans
(32,21)-(32,44)
(32,38)-(32,43)
(32,48)-(32,60)
*)

(* type error slice
(17,26)-(17,50)
(17,27)-(17,43)
(17,28)-(17,40)
(17,44)-(17,45)
(32,21)-(32,44)
(32,22)-(32,34)
(32,47)-(32,70)
(32,48)-(32,60)
*)

(* all spans
(13,21)-(33,53)
(14,2)-(33,53)
(14,8)-(14,9)
(15,13)-(15,16)
(16,13)-(16,16)
(17,14)-(17,50)
(17,24)-(17,25)
(17,14)-(17,23)
(17,26)-(17,50)
(17,44)-(17,45)
(17,27)-(17,43)
(17,28)-(17,40)
(17,41)-(17,42)
(17,46)-(17,49)
(18,16)-(18,52)
(18,26)-(18,27)
(18,16)-(18,25)
(18,28)-(18,52)
(18,46)-(18,47)
(18,29)-(18,45)
(18,30)-(18,42)
(18,43)-(18,44)
(18,48)-(18,51)
(20,6)-(20,71)
(20,11)-(20,12)
(20,6)-(20,10)
(20,13)-(20,71)
(20,32)-(20,33)
(20,14)-(20,31)
(20,15)-(20,27)
(20,28)-(20,30)
(20,34)-(20,70)
(20,39)-(20,40)
(20,35)-(20,38)
(20,41)-(20,69)
(20,60)-(20,61)
(20,42)-(20,59)
(20,43)-(20,55)
(20,56)-(20,58)
(20,62)-(20,68)
(21,21)-(21,66)
(21,39)-(21,40)
(21,21)-(21,38)
(21,22)-(21,34)
(21,35)-(21,37)
(21,41)-(21,66)
(21,46)-(21,47)
(21,42)-(21,45)
(21,48)-(21,65)
(21,49)-(21,61)
(21,62)-(21,64)
(23,6)-(28,79)
(23,10)-(23,11)
(23,6)-(23,9)
(24,8)-(28,79)
(24,27)-(24,28)
(24,9)-(24,26)
(24,10)-(24,22)
(24,23)-(24,25)
(25,11)-(28,78)
(25,16)-(25,17)
(25,12)-(25,15)
(26,14)-(28,77)
(26,33)-(26,34)
(26,15)-(26,32)
(26,16)-(26,28)
(26,29)-(26,31)
(27,17)-(28,76)
(27,22)-(27,23)
(27,18)-(27,21)
(28,20)-(28,75)
(28,39)-(28,40)
(28,21)-(28,38)
(28,22)-(28,34)
(28,35)-(28,37)
(28,41)-(28,74)
(28,46)-(28,47)
(28,42)-(28,45)
(28,48)-(28,73)
(28,67)-(28,68)
(28,49)-(28,66)
(28,50)-(28,62)
(28,63)-(28,65)
(28,69)-(28,72)
(30,6)-(32,73)
(30,10)-(30,11)
(30,6)-(30,9)
(31,8)-(32,73)
(31,27)-(31,28)
(31,9)-(31,26)
(31,10)-(31,22)
(31,23)-(31,25)
(32,11)-(32,72)
(32,18)-(32,19)
(32,12)-(32,17)
(32,20)-(32,71)
(32,45)-(32,46)
(32,21)-(32,44)
(32,22)-(32,34)
(32,35)-(32,37)
(32,38)-(32,43)
(32,47)-(32,70)
(32,48)-(32,60)
(32,61)-(32,63)
(32,64)-(32,69)
(33,19)-(33,53)
(33,26)-(33,27)
(33,19)-(33,25)
(33,28)-(33,53)
(33,47)-(33,48)
(33,29)-(33,46)
(33,30)-(33,42)
(33,43)-(33,45)
(33,49)-(33,52)
*)
