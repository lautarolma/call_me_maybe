EN el caso de las variables PyObj con o sin Slots (__slot__) hemos vistas que estas fuerzan la generacion de una variable/objeto de python lo mas afin a una variable sencilla y simplificada, sin tantos atributos especiales propios de las variables objeto. Lo cual les permite una velocidad de acceso a sus atributos mejorada y optimizacion de uso de memoria.
 **EL caso de las verificaciones segun la naturaleza del dato** 

    Para datos internos: DataClass. Para generar la variable que lo contenga, pues sera un __slot__ data con atributos optimizados, pues no requiere de validacion al estar preestablecido por el sistema sin intervencion de terceros externos.
    Para datos externos (I/O): Pydantic. Ha de pasar por verificacion previa, pues, provenientes de una APIs, un JSON, o imput d eusuario
