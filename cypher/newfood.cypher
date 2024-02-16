CALL apoc.schema.assert(null, {Ingredient:['name'], Dish:['id'], DishType:['name']});

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/smallcat9603/graph/main/data/newfood.csv" AS row 
CREATE (d:Dish {id:row.id}) 
SET d += apoc.map.clean(row, ['id','dishTypes','ingredients'],[]) 
FOREACH (i in split(row.ingredients,',') | MERGE (in:Ingredient {name:toLower(replace(i,'-',' '))}) MERGE (in)<-[:HAS_INGREDIENT]-(d)) 
FOREACH (dt in split(row.dishTypes,',')  | MERGE (dts:DishType {name:dt}) MERGE (dts)<-[:DISH_TYPE]-(d));
